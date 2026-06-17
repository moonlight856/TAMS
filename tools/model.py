"""TAMS model: integrates MATL, DT-GNN, OT-CMA, and AMRG modules."""

from __future__ import annotations

import setup_paths  # noqa: F401

import os
import torch
from torch.nn import Module, ModuleList, Conv1d, Sequential, ReLU, Dropout, GELU
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import Linear, GATv2Conv, GraphNorm

from block import ThresBlock
from modules_m1_input_fusion import VisualOnlyInput, ModalityInputFusion
from modules_m2_ca_gbt import CAThresBlock
from modules_m3_hetero import GatedModalityFusion
from modules_query_bridge import QueryBridge
from modules_query_cross_attn import QueryCrossAttnFusion


class DilatedResidualLayer(Module):
    def __init__(self, dilation, in_channels, out_channels):
        super().__init__()
        self.conv_dilated = Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1x1 = Conv1d(out_channels, out_channels, kernel_size=1)
        self.relu = ReLU()
        self.dropout = Dropout()

    def forward(self, x):
        out = self.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Refinement(Module):
    def __init__(self, final_dim, num_layers=10, interm_dim=64):
        super().__init__()
        self.conv_1x1 = Conv1d(final_dim, interm_dim, kernel_size=1)
        self.layers = ModuleList([DilatedResidualLayer(2**i, interm_dim, interm_dim) for i in range(num_layers)])
        self.conv_out = Conv1d(interm_dim, final_dim, kernel_size=1)

    def forward(self, x):
        f = self.conv_1x1(x)
        for layer in self.layers:
            f = layer(f)
        out = self.conv_out(f)
        return out


class ScoreCalibration(Module):
    """Learnable temperature + bias for frame-level scores.

    Helps knapsack segment selection by sharpening the score distribution.
    Initialized near identity (temperature=1, bias=0) so early training
    is unaffected.
    """

    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return self.temperature * scores + self.bias



def _make_thres(cfg, in_ch, out_ch):
    # Verification-stage switch: keep CA-GBT implementation in codebase,
    # but allow fully bypassing dual-threshold reads when requested.
    # When `disable_dual_threshold` is true, never read tau1/tau2 and always
    # instantiate plain ThresBlock for legacy slots.
    if bool(cfg.get("disable_dual_threshold", False)):
        return ThresBlock(in_ch, out_ch)

    if cfg.get("m2_visual_type") == "ca_gbt":
        return CAThresBlock(
            in_ch,
            out_ch,
            tau_low=float(cfg.get("tau1_init", 0.2)),
            tau_high=float(cfg.get("tau2_init", 0.5)),
        )
    return ThresBlock(in_ch, out_ch)


class TAMS(Module):
    def __init__(self, cfg, t_emb=True):
        super().__init__()
        self.cfg = cfg
        # Compatibility-safe reads: allow slim YAMLs without explicit legacy keys.
        self.use_spf = cfg.get("use_spf", False)
        self.use_ref = cfg.get("use_ref", False)
        self.num_modality = cfg["num_modality"]
        channels = [cfg["channel1"], cfg["channel2"]]
        final_dim = cfg["final_dim"]
        num_att_heads = int(cfg.get("num_att_heads") or 0)
        dropout = cfg["dropout"]
        self.task_mode = cfg.get("task_mode", "generic")
        self.use_query = cfg.get("use_query", False) and self.task_mode == "query"
        qdim = cfg.get("query_in_dim", channels[0])

        if self.use_spf:
            self.layer_spf = Linear(-1, cfg["proj_dim"])

        self.layer011 = Linear(-1, channels[0])
        if self.num_modality >= 2:
            self.layer012 = Linear(-1, channels[0])
        if self.num_modality >= 3:
            self.linear_mod3 = Linear(-1, channels[0])

        self.batch01 = GraphNorm(channels[0])
        self.relu = GELU()
        self.dropout = Dropout(dropout)

        self.norm = cfg["norm"]
        _n_il = int(cfg.get("interleaved_layers", 0))

        if _n_il == 0:
            self.layer11 = _make_thres(cfg, channels[0], channels[0])
            self.batch11 = GraphNorm(channels[0])
            self.layer12 = _make_thres(cfg, channels[0], channels[0])
            self.batch12 = GraphNorm(channels[0])
            self.layer13 = _make_thres(cfg, channels[0], channels[0])
            self.batch13 = GraphNorm(channels[0])

            if num_att_heads > 0:
                self.layer21 = GATv2Conv(channels[0], channels[1], heads=num_att_heads)
            else:
                self.layer21 = _make_thres(cfg, channels[0], channels[1])
                num_att_heads = 1
            self.batch21 = GraphNorm(channels[1] * num_att_heads)

            self.layer31 = _make_thres(cfg, channels[1] * num_att_heads, final_dim)
            self.layer32 = _make_thres(cfg, channels[1] * num_att_heads, final_dim)
            self.layer33 = _make_thres(cfg, channels[1] * num_att_heads, final_dim)

            if self.use_ref:
                self.layer_ref1 = Refinement(final_dim)
                self.layer_ref2 = Refinement(final_dim)
                self.layer_ref3 = Refinement(final_dim)
        else:
            num_att_heads = max(num_att_heads, 1)

        self.t_emb = torch.nn.Parameter(torch.zeros(2000, channels[0])) if t_emb else None

        self.query_bridge = None
        self.query_cross = None
        if self.use_query:
            qfus = cfg.get("query_fusion", "bridge")
            if qfus == "cross_attn":
                heads = int(cfg.get("query_cross_attn_heads", 4))
                self.query_cross = QueryCrossAttnFusion(
                    channels[0], qdim, n_heads=heads, dropout=float(cfg.get("query_cross_attn_dropout", 0.1))
                )
            else:
                self.query_bridge = QueryBridge(channels[0], query_dim=qdim)

        self.plan_wire_m1 = cfg.get("plan_wire_m1", False)
        self.m1_use_missing_token = cfg.get("m1_use_missing_token", False)
        self.m1_vis = None
        self.m1_fusion = None
        if self.m1_use_missing_token:
            self.m1_fusion = ModalityInputFusion(channels[0], n_modalities=self.num_modality)
        elif self.plan_wire_m1:
            self.m1_vis = VisualOnlyInput()

        self.plan_wire_m3 = cfg.get("plan_wire_m3", False)
        self.m3_fusion = None
        if self.plan_wire_m3 and self.num_modality > 1:
            ft = cfg.get("fusion_type")
            if ft == "gated":
                self.m3_fusion = GatedModalityFusion(channels[0], n_modalities=self.num_modality)
            elif ft == "concat_proj":
                from modules_m3_hetero import ConcatLinearFusion

                self.m3_fusion = ConcatLinearFusion(channels[0], self.num_modality, dropout=dropout)

        md = cfg.get("videoxum_modality_dim")
        modality_dims = cfg.get("modality_dims")
        self.text_sage = None
        self.audio_sage = None
        txt_sage_dim = modality_dims[1] if modality_dims and len(modality_dims) >= 2 else md
        aud_sage_dim = modality_dims[2] if modality_dims and len(modality_dims) >= 3 else md
        if isinstance(txt_sage_dim, int) and txt_sage_dim > 0:
            if cfg.get("use_text") and cfg.get("m2_text_type") == "sage" and self.num_modality >= 2:
                from modules_m2_text import TemporalShotSAGE

                self.text_sage = TemporalShotSAGE(txt_sage_dim)
        if isinstance(aud_sage_dim, int) and aud_sage_dim > 0:
            if cfg.get("use_audio") and cfg.get("m2_audio_type") == "sage" and self.num_modality >= 3:
                from modules_m2_audio import TemporalAudioSAGE

                self.audio_sage = TemporalAudioSAGE(aud_sage_dim)

        self.use_m4_head = False

        self.n_interleaved = int(cfg.get("interleaved_layers", 0))
        self._ms_taus = cfg.get("multi_scale_taus") or []
        self.score_head = None

        self.use_matl = cfg.get("use_matl", False)
        self.use_ta_gnn = cfg.get("use_ta_gnn", False)
        self.use_ot_cma = cfg.get("use_ot_cma", False)
        self.use_directional_gnn = cfg.get("use_directional_gnn", False)
        self.use_modality_gate = cfg.get("use_modality_gate", False)

        self.matl = None
        self.ta_gnn_layers = None
        self.ot_cma_layers = None
        self.dir_gnn_layers = None
        self.modality_gate = None

        if self.n_interleaved > 0:
            ch = channels[0]

            if self.use_modality_gate and self.num_modality > 1:
                from modules_ot_cma import AdaptiveModalityRelevanceGate

                self.modality_gate = AdaptiveModalityRelevanceGate(self.num_modality)

            if self.use_matl:
                from modules_topology_learner import ModalityAdaptiveTopologyLearner

                matl_scales = cfg.get("matl_scales") or [3, 10, 30]
                self.matl = ModalityAdaptiveTopologyLearner(
                    dim=ch,
                    n_modalities=self.num_modality,
                    scales=matl_scales,
                    max_rel_pos=cfg.get("matl_max_rel_pos"),
                    temperature_init=float(cfg.get("matl_temperature_init", 1.0)),
                    shared_topology=bool(cfg.get("matl_shared_topology", False)),
                )

            if self.use_directional_gnn:
                from modules_ta_gnn import DirectionalTopologyGNN

                self.dir_gnn_layers = ModuleList([
                    DirectionalTopologyGNN(ch, ch)
                    for _ in range(self.n_interleaved)
                ])
            elif self.use_ta_gnn:
                from modules_ta_gnn import TopologyAwareMessagePassing

                self.ta_gnn_layers = ModuleList([
                    TopologyAwareMessagePassing(ch, ch)
                    for _ in range(self.n_interleaved)
                ])

            if self.use_ot_cma:
                from modules_ot_cma import OTCrossModalAlignment

                self.ot_cma_layers = ModuleList([
                    OTCrossModalAlignment(
                        ch,
                        n_modalities=self.num_modality,
                        sinkhorn_reg=float(cfg.get("sinkhorn_reg", 0.1)),
                        sinkhorn_iter=int(cfg.get("sinkhorn_iter", 8)),
                        topo_weight=float(cfg.get("ot_cma_topo_weight", 0.1)),
                        dropout=float(cfg.get("ot_cma_dropout", 0.1)),
                        topo_stop_grad=bool(cfg.get("ot_cma_topo_stop_grad", False)),
                    )
                    for _ in range(self.n_interleaved)
                ])

            if not self.use_ta_gnn and not self.use_directional_gnn:
                self.interleaved_gnn = ModuleList([
                    _make_thres(cfg, ch, ch) for _ in range(self.n_interleaved)
                ])
            self.interleaved_gnn_norm = ModuleList([
                GraphNorm(ch) for _ in range(self.n_interleaved)
            ])

            if not self.use_ot_cma:
                from modules_m3_hetero import CrossModalFusionBlock

                cmf_heads = int(cfg.get("cmf_n_heads", 4))
                self.interleaved_cmf = ModuleList([
                    CrossModalFusionBlock(
                        ch, n_heads=cmf_heads,
                        n_modalities=self.num_modality, dropout=dropout,
                    )
                    for _ in range(self.n_interleaved)
                ])

            self.score_head = Sequential(
                torch.nn.LayerNorm(ch),
                Linear(ch, ch // 2), GELU(), Dropout(dropout),
                Linear(ch // 2, ch // 4), GELU(),
                Linear(ch // 4, 1),
            )

            self.fused_scale = torch.nn.Parameter(torch.tensor(0.3))

            self.score_calibration = None
            if cfg.get("use_score_calibration", False):
                self.score_calibration = ScoreCalibration()

    def _conv(self, layer, x, edge_index, node_conf):
        if isinstance(layer, CAThresBlock):
            return layer(x, edge_index, node_conf)
        return layer(x, edge_index)

    def _full_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        c,
        query_emb,
        node_conf: torch.Tensor,
        modality_scale: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        nm = int(self.num_modality)
        modality_dims = self.cfg.get("modality_dims")
        if modality_dims and len(modality_dims) >= nm:
            cum = 0
            xv_raw = x[:, cum : cum + modality_dims[0]]
            cum += modality_dims[0]
            xt_raw = x[:, cum : cum + modality_dims[1]] if nm >= 2 else None
            cum += modality_dims[1] if nm >= 2 else 0
            xa_raw = x[:, cum : cum + modality_dims[2]] if nm >= 3 else None
        else:
            feature_dim = x.shape[1]
            if feature_dim % nm != 0:
                raise ValueError(f"feature_dim {feature_dim} not divisible by num_modality {nm}")
            chunk = feature_dim // nm
            xv_raw = x[:, :chunk]
            xt_raw = x[:, chunk : 2 * chunk] if nm >= 2 else None
            xa_raw = x[:, 2 * chunk : 3 * chunk] if nm >= 3 else None

        if self.use_spf:
            x_visual = self.layer011(torch.cat((xv_raw, self.layer_spf(c)), dim=1))
        else:
            x_visual = self.layer011(xv_raw)

        if self.m1_vis is not None:
            x_visual = self.m1_vis(x_visual)

        h_parts: list[torch.Tensor] = [x_visual]
        if nm >= 2 and xt_raw is not None:
            t_in = xt_raw
            if self.text_sage is not None:
                t_in = self.text_sage(t_in, edge_index)
            h_parts.append(self.layer012(t_in))
        if nm >= 3 and xa_raw is not None:
            a_in = xa_raw
            if self.audio_sage is not None:
                a_in = self.audio_sage(a_in, edge_index)
            h_parts.append(self.linear_mod3(a_in))

        if self.m1_fusion is not None:
            h_parts = self.m1_fusion(h_parts, modality_scale)
        elif modality_scale is not None and len(modality_scale) == len(h_parts):
            scaled: list[torch.Tensor] = []
            for hp, s in zip(h_parts, modality_scale):
                sv = s.to(device=hp.device, dtype=hp.dtype).clamp(0.0, 1.0)
                if sv.shape[0] != hp.shape[0]:
                    raise ValueError(
                        f"modality_scale length {sv.shape[0]} != h_part nodes {hp.shape[0]}"
                    )
                scaled.append(hp * sv.unsqueeze(-1))
            h_parts = scaled

        h_parts_pre_fusion = h_parts

        if self.n_interleaved > 0:
            # Apply query conditioning before interleaved loop (QFVS support)
            if self.use_query and query_emb is not None:
                qe = query_emb.to(h_parts[0].device)
                for i in range(len(h_parts)):
                    if self.query_cross is not None:
                        h_parts[i] = self.query_cross(h_parts[i], qe)
                    elif self.query_bridge is not None:
                        h_parts[i] = self.query_bridge(h_parts[i], qe)
            if self.use_matl or self.use_ta_gnn or self.use_ot_cma or self.use_directional_gnn:
                return self._interleaved_forward_v2(
                    h_parts, edge_index, node_conf, modality_scale,
                )
            return self._interleaved_forward(h_parts, edge_index, node_conf)

        if len(h_parts) == 1:
            h = h_parts[0]
        elif self.m3_fusion is not None:
            h = self.m3_fusion(*h_parts)
        else:
            h = h_parts[0]
            for t in h_parts[1:]:
                h = h + t

        if self.norm:
            h = self.batch01(h)
        h = self.relu(h)

        if self.use_query and query_emb is not None:
            qe = query_emb.to(h.device)
            if self.query_cross is not None:
                h = self.query_cross(h, qe)
            elif self.query_bridge is not None:
                h = self.query_bridge(h, qe)

        edge_index_f = edge_index[:, edge_attr <= 0]
        edge_index_b = edge_index[:, edge_attr >= 0]

        x1 = self._conv(self.layer11, h, edge_index_f, node_conf)
        if self.norm:
            x1 = self.batch11(x1)
        x1 = self.relu(x1)
        if self.t_emb is not None:
            x1 = x1 + self.t_emb[: x1.shape[0]]
        x1 = self.dropout(x1)
        x1 = self._conv(self.layer21, x1, edge_index_f, node_conf)
        if self.norm:
            x1 = self.batch21(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        x2 = self._conv(self.layer12, h, edge_index_b, node_conf)
        if self.norm:
            x2 = self.batch12(x2)
        x2 = self.relu(x2)
        if self.t_emb is not None:
            x2 = x2 + self.t_emb[: x2.shape[0]]
        x2 = self.dropout(x2)
        x2 = self._conv(self.layer21, x2, edge_index_b, node_conf)
        if self.norm:
            x2 = self.batch21(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)

        x3 = self._conv(self.layer13, h, edge_index, node_conf)
        if self.norm:
            x3 = self.batch13(x3)
        x3 = self.relu(x3)
        if self.t_emb is not None:
            x3 = x3 + self.t_emb[: x3.shape[0]]
        x3 = self.dropout(x3)
        x3 = self._conv(self.layer21, x3, edge_index, node_conf)
        if self.norm:
            x3 = self.batch21(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)

        x1 = self._conv(self.layer31, x1, edge_index_f, node_conf)
        x2 = self._conv(self.layer32, x2, edge_index_b, node_conf)
        x3 = self._conv(self.layer33, x3, edge_index, node_conf)

        out = x1 + x2 + x3
        self._last_gate_alpha = None

        if self.use_ref:
            xr0 = torch.permute(out, (1, 0)).unsqueeze(0)
            xr1 = self.layer_ref1(torch.softmax(xr0, dim=1))
            xr2 = self.layer_ref2(torch.softmax(xr1, dim=1))
            xr3 = self.layer_ref3(torch.softmax(xr2, dim=1))
            out = torch.stack((xr0, xr1, xr2, xr3), dim=0).squeeze(1).transpose(2, 1).contiguous()

        return out

    @staticmethod
    def _filter_edges_by_tau(edge_index: torch.Tensor, tau: int) -> torch.Tensor:
        """Keep only edges where |src - dst| <= tau for multi-scale filtering."""
        dist = (edge_index[0] - edge_index[1]).abs()
        return edge_index[:, dist <= tau]

    @staticmethod
    @torch.no_grad()
    def _soft_edge_weights(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Content-based soft edge weights via cosine similarity.

        When MATL is off, this replaces the all-ones fallback so that
        message passing is content-aware rather than uniform aggregation.
        Detached from the computation graph to avoid extra gradient paths
        through the similarity computation (the GNN layers themselves
        still receive gradients through the message MLP + shortcut).
        """
        src, dst = edge_index[0], edge_index[1]
        h_f = h.float()
        h_src = h_f[src]
        h_dst = h_f[dst]
        norm_s = h_src.norm(dim=1).clamp(min=1e-6)
        norm_d = h_dst.norm(dim=1).clamp(min=1e-6)
        sim = (h_src * h_dst).sum(dim=1) / (norm_s * norm_d)
        return torch.sigmoid(sim * 4.0).to(dtype=h.dtype)

    def _interleaved_forward(
        self,
        h_parts: list[torch.Tensor],
        edge_index: torch.Tensor,
        node_conf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interleaved multi-scale GNN + cross-modal fusion (legacy path).

        Each layer l:
          1. Per-modality CA-GBT at temporal scale tau_l  (shared across modalities)
          2. CrossModalFusion: fusion token queries modality features
        Fusion outputs are residually accumulated across layers so every
        temporal scale contributes to the final prediction.
        """
        h_fused = torch.zeros_like(h_parts[0])
        for l in range(self.n_interleaved):
            tau_l = self._ms_taus[l] if l < len(self._ms_taus) else (
                self._ms_taus[-1] if self._ms_taus else 999
            )
            edge_l = self._filter_edges_by_tau(edge_index, tau_l)
            for i in range(len(h_parts)):
                h = self._conv(self.interleaved_gnn[l], h_parts[i], edge_l, node_conf)
                h = self.interleaved_gnn_norm[l](h)
                h = self.relu(h)
                if self.t_emb is not None:
                    h = h + self.t_emb[: h.shape[0]]
                h_parts[i] = self.dropout(h) + h_parts[i]
            h_fused = h_fused + self.interleaved_cmf[l](*h_parts)
        self._last_gate_alpha = None
        return self.score_head(h_fused)

    def _interleaved_forward_v2(
        self,
        h_parts: list[torch.Tensor],
        edge_index: torch.Tensor,
        node_conf: torch.Tensor,
        modality_scale: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        TAMS interleaved forward (replaces _interleaved_forward when v2 modules active).

        Each layer l:
          0. (once) AMRG: adaptive modality relevance gate
          1. MATL: per-modality topology learning with Gumbel-Sigmoid
          2. DT-GNN / TA-GNN: directional or shared message passing
          3. OT-CMA / TBCMI / CMF: cross-modal fusion
        After all layers, optionally apply M4 confidence gating for multi-modal configs.
        """
        h_fused = torch.zeros_like(h_parts[0])
        all_weights: list[list[torch.Tensor]] = []
        all_adjs: list[list[torch.Tensor]] = []

        node_conf_list: list[torch.Tensor] | None = None
        if modality_scale is not None and len(modality_scale) > 0:
            node_conf_list = modality_scale

        # --- Step 0: Adaptive Modality Relevance Gate ---
        if self.modality_gate is not None:
            h_parts, gate_vals = self.modality_gate(h_parts)
            self._last_modality_gate = gate_vals
        else:
            self._last_modality_gate = None

        for l in range(self.n_interleaved):
            # --- Step 1: Topology learning ---
            if self.matl is not None:
                adj_list, weight_list = self.matl(h_parts, scale_idx=l, node_conf=node_conf_list)
                all_weights.append(weight_list)
                all_adjs.append(adj_list)
            else:
                tau_l = self._ms_taus[l] if l < len(self._ms_taus) else (
                    self._ms_taus[-1] if self._ms_taus else 999
                )
                edge_l = self._filter_edges_by_tau(edge_index, tau_l)
                adj_list = [edge_l for _ in range(len(h_parts))]
                weight_list = [
                    self._soft_edge_weights(h_parts[m], edge_l)
                    for m in range(len(h_parts))
                ]
                all_adjs.append(adj_list)

            # --- Step 2: Per-modality GNN ---
            # Add t_emb only at the first interleaved layer (avoid 9x accumulation
            # over 3 layers x 3 modalities which used to dominate features).
            add_temb_here = (self.t_emb is not None) and (l == 0)
            for m in range(len(h_parts)):
                edge_m = adj_list[m]
                w_m = weight_list[m]

                if self.dir_gnn_layers is not None:
                    h = self.dir_gnn_layers[l](h_parts[m], edge_m, w_m)
                elif self.ta_gnn_layers is not None:
                    mask_f = edge_m[0] > edge_m[1]
                    mask_b = edge_m[0] < edge_m[1]
                    h_f = self.ta_gnn_layers[l](h_parts[m], edge_m[:, mask_f], w_m[mask_f])
                    h_b = self.ta_gnn_layers[l](h_parts[m], edge_m[:, mask_b], w_m[mask_b])
                    h_u = self.ta_gnn_layers[l](h_parts[m], edge_m, w_m)
                    h = h_f + h_b + h_u
                elif hasattr(self, 'interleaved_gnn'):
                    mask_f = edge_m[0] > edge_m[1]
                    mask_b = edge_m[0] < edge_m[1]
                    h_f = self._conv(self.interleaved_gnn[l], h_parts[m], edge_m[:, mask_f], node_conf)
                    h_b = self._conv(self.interleaved_gnn[l], h_parts[m], edge_m[:, mask_b], node_conf)
                    h_u = self._conv(self.interleaved_gnn[l], h_parts[m], edge_m, node_conf)
                    h = h_f + h_b + h_u
                else:
                    h = h_parts[m]

                h = self.interleaved_gnn_norm[l](h)
                h = self.relu(h)
                if add_temb_here:
                    h = h + self.t_emb[: h.shape[0]]
                h_parts[m] = self.dropout(h) + h_parts[m]

            # --- Step 3: Cross-modal fusion ---
            if self.ot_cma_layers is not None:
                h_fused = h_fused + self.ot_cma_layers[l](h_parts, adj_list, weight_list)
            elif hasattr(self, 'interleaved_cmf'):
                h_fused = h_fused + self.interleaved_cmf[l](*h_parts)
            else:
                h_fused = h_fused + sum(h_parts) / len(h_parts)

        self._last_all_weights = all_weights
        self._last_all_adjs = all_adjs

        h_modality_avg = sum(h_parts) / len(h_parts)
        scale = self.fused_scale.clamp(0.01, 2.0)
        h_final = scale * h_fused + h_modality_avg

        self._last_gate_alpha = None
        score = self.score_head(h_final)

        if self.score_calibration is not None:
            score = self.score_calibration(score)

        return score

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        c=None,
        query_emb=None,
        node_conf=None,
        modality_scale: list[torch.Tensor] | None = None,
    ):
        n = x.shape[0]
        if node_conf is None:
            if self.cfg.get("use_step0"):
                from modules_step0 import default_visual_confidence

                node_conf = default_visual_confidence(n, x.device, x.dtype)
            else:
                node_conf = torch.ones(n, device=x.device, dtype=x.dtype)
        else:
            node_conf = node_conf.to(device=x.device, dtype=x.dtype)

        if self.training and self.cfg.get("gradient_checkpointing", False):
            return checkpoint(
                self._full_forward,
                x,
                edge_index,
                edge_attr,
                c,
                query_emb,
                node_conf,
                modality_scale,
                use_reentrant=False,
            )
        return self._full_forward(
            x, edge_index, edge_attr, c, query_emb, node_conf, modality_scale,
        )


def build_model(cfg):
    from gravit.utils.cfg_defaults import merge_defaults

    merge_defaults(cfg)
    name = cfg.get("model_name", "TAMS")
    if name == "TAMS":
        return TAMS(cfg, cfg.get("t_emb", True))
    raise ValueError(f"Unknown model_name: {name}")


def build_dataloaders(cfg, path_graphs_root: str):
    from torch_geometric.loader import DataLoader
    from gravit.datasets import GraphDataset

    from gravit.utils.cfg_defaults import merge_defaults

    merge_defaults(cfg)
    if cfg.get("dataset") == "QFVS":
        from gravit.datasets.qfvs_dataset import QFVSDataset

        nw = int(cfg.get("qfvs_dataloader_num_workers", 0))
        pm = bool(cfg.get("qfvs_dataloader_pin_memory", True)) and torch.cuda.is_available()
        train_loader = DataLoader(
            QFVSDataset(os.path.join(path_graphs_root, "train")),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=nw,
            pin_memory=pm,
            persistent_workers=nw > 0,
        )
        val_loader = DataLoader(
            QFVSDataset(os.path.join(path_graphs_root, "val")),
            num_workers=nw,
            pin_memory=pm,
            persistent_workers=nw > 0,
        )
        return train_loader, val_loader

    train_dir = os.path.join(path_graphs_root, "train")
    val_dir = os.path.join(path_graphs_root, "val")
    train_ds = GraphDataset(train_dir)
    val_ds = GraphDataset(val_dir)
    if len(train_ds) == 0:
        gn = cfg.get("graph_name", "<graph_name>")
        ds_name = cfg.get("dataset", "SumMe")
        raise FileNotFoundError(
            "Training set is empty: no *.pt graphs in:\n"
            f"  {os.path.abspath(train_dir)}\n"
            f"With graph_name '{gn}', data must live under "
            f"<root_data>/graphs/{gn}/split<N>/train/ (N from --split or --all_splits).\n"
            "Build SumMe/TVSum graphs with:\n"
            f"  python data/generate_temporal_graphs.py --dataset {ds_name} "
            "--features <H5_BASENAME> --tauf T --skip_factor S\n"
            "Output goes to graphs/{dataset}_T_S/split1..5/; T and S must match your "
            f"graph_name (e.g. TVSum_10_0 → tauf=10, skip_factor=0). "
            "See README.md."
        )
    if len(val_ds) == 0:
        raise FileNotFoundError(
            "Validation set is empty: no *.pt graphs in:\n"
            f"  {os.path.abspath(val_dir)}"
        )

    nw = int(cfg.get("stvs_dataloader_num_workers", 0))
    pm = bool(cfg.get("stvs_dataloader_pin_memory", True)) and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=nw > 0,
    )
    return train_loader, val_loader
