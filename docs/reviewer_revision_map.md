# WF-IoT camera-ready: sostituzioni testuali 1–13

> **Regola scientifica:** i valori indicati come `TBD` vanno sostituiti esclusivamente con output reali del nuovo protocollo. Le stime sono fornite separatamente in `results/provisional_expected_ranges.csv` e **non devono essere presentate come risultati misurati**.

## 1. Abstract
### Sostituire
La parte da: `Evaluation is performed with a leave-one-node-out protocol...` fino alla fine dell'abstract.

### Con
```latex
Evaluation follows a leakage-controlled cross-node protocol in which the outer test node is kept completely unseen during training, scaling, early stopping, and threshold calibration. A complete five-fold leave-one-node-out (LONO) cycle is used to quantify cross-node generalization, while repeated runs characterize training variability. In addition to predictive performance, we report computational cost, model memory footprint, and gateway-edge inference latency/energy, and we assess the sensitivity of the warning behavior to the prediction horizon. Under the reference horizon $H=15$, FT-64 achieves a macro-averaged AUC-ROC of \textbf{TBD} across the five held-out nodes, with a recall of \textbf{TBD} and an early-warning coverage of \textbf{TBD}. The model requires approximately 7.52 MMAC per inference and 431 KiB of FP32 weights; measured Raspberry Pi 5 latency and energy are \textbf{TBD} ms and \textbf{TBD} mJ/inference, respectively. These results support compact attention-based early warning at the IoT gateway while making the accuracy--anticipation--resource trade-off explicit.
```

## 2. Contribution 3
### Sostituire
```latex
\item a leakage-aware evaluation protocol based on leave-one-node-out validation, multi-run reporting, threshold optimization, and comparison with Bidirectional LSTM (BiLSTM) baselines under both standard and symmetric loss configurations.
```
### Con
```latex
\item a leakage-controlled full five-fold leave-one-node-out protocol in which the outer test node is excluded from scaling, early stopping, threshold calibration, and hyperparameter selection; acquisition-file grouping is used for inner model selection to prevent overlapping windows from crossing partitions.
```

## 3. Contribution 4
### Sostituire
```latex
\item an edge-oriented analysis showing that medium and compact configurations can reach high AUC-ROC with parameter counts compatible with gateway-class and future embedded deployments.
```
### Con
```latex
\item an edge-oriented assessment combining predictive performance with analytical MAC count, weight memory, measured batch-1 gateway latency and energy per inference, together with a sensitivity analysis of the future warning horizon $H$.
```

## 4. Section II-B -- aggiungere benchmarking edge
### Inserire alla fine di `Time-Series Learning for Edge IoT`
```latex
For edge deployment, predictive quality alone is insufficient. TinyML benchmarking practice jointly considers accuracy, latency, and energy because resource-constrained inference is governed by the interaction between model quality and hardware cost \cite{banbury2021mlperf}. This motivates reporting model size and analytical computation together with device-level timing and energy measurements rather than inferring deployability from parameter count alone. Related BME688-oriented smoke-detection studies also show the relevance of edge execution and explainability for low-power environmental sensing \cite{lehnert2024xplainable}.
```

### Sempre in Section II, aggiungere disclosure MeditCom
```latex
A preliminary version of FireTransformer was presented at IEEE MeditCom 2026 \cite{matta2026firetransformer}. That study established the feasibility of an encoder-only Transformer for BME688-based wildfire detection. The present WF-IoT work extends it by reformulating detection as horizon-based warning, introducing leakage-controlled cross-node evaluation, repeated-run stability analysis, horizon sensitivity, and explicit computational/energy characterization for edge deployment. The dataset used in the present study is publicly available in Mendeley Data \cite{anedda2026dataset}.
```

## 5. Section V-A -- Dataset and Split
### Sostituire l'intera subsection con
```latex
\subsection{Dataset and Leakage-Controlled Cross-Node Split}
The dataset contains 107,493 valid samples after removing initialization and acquisition-error records. Fire samples account for 92,395 records (85.95\%), while normal samples account for 15,098 records (14.05\%). Sequence generation is performed independently within each acquisition file. The underlying public dataset is described in \cite{anedda2026dataset}.

To evaluate deployment on previously unseen sensing hardware, we use a complete five-fold leave-one-node-out (LONO) protocol. At outer fold $k$, all acquisitions of node $k$ are reserved as the test set and remain completely unavailable during scaler fitting, training, early stopping, threshold calibration, and hyperparameter selection. The remaining four nodes form the development set. Inner training and validation partitions are created at acquisition-file level, so that all sliding windows originating from the same chronological acquisition remain in the same partition. This prevents near-duplicate overlapping windows from crossing model-selection boundaries.

For each outer fold, Min--Max scaling is fitted using the inner training partition only and subsequently applied unchanged to the inner validation and outer test data. Early stopping and the decision threshold $\tau$ are determined exclusively on the inner validation data. The selected model and threshold are then evaluated once on the untouched outer node. Results are reported per held-out node and as macro averages across the five outer folds; repeated random seeds quantify optimization variability.
```

## 6. Section V-C -- Metrics
### Sostituire l'intera subsection con
```latex
\subsection{Metrics and Edge-Cost Measures}
Predictive performance is quantified using precision, recall, F1-score, AUC-ROC, early-warning coverage, and timestamp-based lead time. AUC-ROC characterizes threshold-independent discrimination, while recall and onset coverage emphasize missed-event risk. Lead time is computed as the difference between the annotated physical onset and the end timestamp of the earliest correctly warning window associated with that onset.

To evaluate edge feasibility, we additionally report the number of trainable parameters, FP32 weight footprint, multiply--accumulate operations (MACs) per batch-1 inference, median and 95th-percentile inference latency, average active power, and energy per inference. Following the benchmarking principles of \cite{banbury2021mlperf}, latency and energy are measured on the same hardware/software configuration after a warm-up phase. System energy is computed as $E_{\mathrm{sys}}=P_{\mathrm{active}}t_{\mathrm{inf}}$, while incremental inference energy is $E_{\mathrm{dyn}}=(P_{\mathrm{active}}-P_{\mathrm{idle}})t_{\mathrm{inf}}$.
```

## 7. Section VI-A -- Full LONO
### Sostituire titolo e testo iniziale con
```latex
\subsection{Full Leave-One-Node-Out Generalization}
Table~\ref{tab:lono} reports test performance when each physical sensing node is held out in turn. Unlike the original single-node split, the outer node is not involved in any model-selection decision. The macro average therefore measures zero-shot transfer across physical nodes rather than performance on a validation node used during development. FT-64 achieves a macro AUC-ROC of \textbf{TBD} and macro F1-score of \textbf{TBD}; the between-node standard deviation of AUC is \textbf{TBD}. These results show that the model's cross-node behavior is \textbf{TBD based on measured results}, while also revealing the variability hidden by a single NODO5 holdout.
```

### Nuova tabella
```latex
\begin{table}[t]
\centering
\caption{Full leave-one-node-out performance of FT-64. Values are outer-test results; the held-out node is never used for model selection.}
\label{tab:lono}
\scriptsize
\begin{tabular}{lccccc}
\toprule
Held-out node & Precision & Recall & F1 & AUC & Coverage \\
\midrule
NODO1 & TBD & TBD & TBD & TBD & TBD \\
NODO2 & TBD & TBD & TBD & TBD & TBD \\
NODO3 & TBD & TBD & TBD & TBD & TBD \\
NODO4 & TBD & TBD & TBD & TBD & TBD \\
NODO5 & TBD & TBD & TBD & TBD & TBD \\
\midrule
Macro mean $\pm$ std & TBD & TBD & TBD & TBD & TBD \\
\bottomrule
\end{tabular}
\end{table}
```

## 8. Section VI-B -- Horizon sensitivity
### Sostituire il paragrafo interpretativo sul lead time con
```latex
The approximately 2-s lead time obtained at the reference horizon should be interpreted as \emph{sensor-level anticipation} rather than as an evacuation-scale warning interval. It quantifies how early the local chemical/environmental stream becomes predictive relative to the annotated onset at the sensor. End-to-end operational warning time additionally depends on plume transport, node placement, communication delay, alarm persistence, and escalation policy.

To quantify the effect of the design horizon, FT-64 is evaluated for $H\in\{5,15,30,60\}$ using the same leakage-controlled protocol. Increasing $H$ moves the positive target farther before onset and can increase potential anticipation, but it also weakens precursor separability. Table~\ref{tab:horizon} therefore reports both classification quality and realized lead time rather than assuming a linear gain with $H$. The measured results indicate \textbf{TBD}.
```

### Aggiungere tabella
```latex
\begin{table}[t]
\centering
\caption{Sensitivity of FT-64 to prediction horizon $H$.}
\label{tab:horizon}
\scriptsize
\begin{tabular}{ccccc}
\toprule
$H$ & AUC & Recall & Coverage & Lead time [s] \\
\midrule
5  & TBD & TBD & TBD & TBD \\
15 & TBD & TBD & TBD & TBD \\
30 & TBD & TBD & TBD & TBD \\
60 & TBD & TBD & TBD & TBD \\
\bottomrule
\end{tabular}
\end{table}
```

## 9. Section VI-C -- Computational complexity and energy
### Sostituire titolo `Model Size and Edge Implications` e relativi paragrafi con
```latex
\subsection{Computational Complexity and Edge Energy Cost}
For sequence length $S=W+1=61$, one Transformer encoder layer requires approximately
\begin{equation}
C_{\mathrm{layer}}\!\simeq\!4Sd^2+2Sdd_{\mathrm{ff}}+2S^2d
\end{equation}
MACs, excluding element-wise operations such as LayerNorm, activation functions, and softmax. Including input projection, temporal pooling, and classification, FT-32, FT-64, and FT-128 require approximately 1.52, 7.52, and 27.05 MMAC per inference, respectively. Their FP32 weight footprints are approximately 78.1 KiB, 431.0 KiB, and 1.65 MiB.

Device-level measurements are performed on a Raspberry Pi 5 using batch size one, inference mode, fixed software configuration, and repeated warm-up/timed runs. FT-64 reaches a median latency of \textbf{TBD} ms (95th percentile: \textbf{TBD} ms), with active power of \textbf{TBD} W and measured system energy of \textbf{TBD} mJ/inference. The corresponding dynamic energy above idle is \textbf{TBD} mJ/inference. These values are several orders of magnitude below the sensing interval and support gateway-edge deployment. FT-32 remains the main candidate for future quantized node-edge execution; however, microcontroller suitability is not claimed until integer-kernel, activation-memory, and on-device energy measurements are completed.
```

### Sostituire Table III con
```latex
\begin{table}[t]
\centering
\caption{Model-size and computational-cost characterization.}
\label{tab:edgecost}
\scriptsize
\begin{tabular}{lrrrr}
\toprule
Model & Params & FP32 weights & MMAC & Pi5 energy \\
\midrule
FT-32  & 20,002  & 78.1 KiB  & 1.52  & TBD \\
FT-64  & 110,338 & 431.0 KiB & 7.52  & TBD \\
FT-128 & 433,666 & 1.65 MiB  & 27.05 & TBD \\
\bottomrule
\end{tabular}
\end{table}
```

## 10. Section VI-D -- Stability
### Sostituire
```latex
The lower variance of the Transformer models is consistent with the architectural difference between global attention and recurrent propagation. Since each time step can directly attend to every other time step, the model does not rely on a single recurrent state to carry information through the entire window. In the considered dataset, this appears to reduce sensitivity to weight initialization and data-loader ordering.
```
### Con
```latex
The Transformer configurations exhibit lower run-to-run variance than the recurrent baselines in the present experiment. This empirical result supports repeatability after retraining, but it does not establish self-attention as the causal source of the variance reduction. Differences in optimization geometry, regularization, and model capacity may also contribute and require dedicated controlled experiments to separate.
```

## 11. Sections VI-E/VI-F -- deployment wording
### Sostituire in `Deployment Blueprint`
```latex
In a gateway-edge mode, the Transformer runs on a Raspberry Pi-class node co-located with the LoRa gateway. This is the most realistic near-term option because it preserves real-time autonomy while keeping the sensor nodes simple.
```
### Con
```latex
In the gateway-edge mode, the Transformer runs on a single-board computer co-located with the LoRa gateway. This is the experimentally supported near-term option because it preserves local autonomy while keeping the sensing nodes simple; the measured latency and energy results in Section~VI-C quantify its computational overhead.
```

### Sostituire wording explainability
Sostituire `providing operators with a lightweight explanation` con:
```latex
providing operators with a lightweight temporal diagnostic profile; these weights are not interpreted as a causal explanation of the alert.
```

## 12. Limitations
### Sostituire l'intera subsection con
```latex
\subsection{Limitations}
The full LONO protocol quantifies transfer across the five available sensing nodes, but the number of physical nodes remains limited and all acquisitions originate from the same broader experimental campaign. The dataset was collected outdoors under controlled combustion conditions; uncontrolled wildfires may introduce stronger wind-driven dispersion, fuel heterogeneity, sensor aging/drift, and communication impairments. The horizon analysis quantifies the trade-off within the tested range, but the operationally optimal $H$ depends on sampling rate, node-to-source distance, and alarm policy. Finally, Raspberry Pi measurements support gateway-edge feasibility, whereas direct node-edge deployment of FT-32 still requires quantization-aware validation, peak activation-memory profiling, and energy measurements on the target microcontroller.
```

## 13. Conclusion
### Sostituire l'intera conclusione con
```latex
\section{Conclusion}
This paper presented FireTransformer, a compact encoder-only Transformer for horizon-based wildfire warning from distributed BME688 IoT sensing streams. Compared with the preliminary detection-oriented study \cite{matta2026firetransformer}, the present work focuses on anticipatory warning and strengthens deployment-oriented validation through a leakage-controlled full five-fold leave-one-node-out protocol, repeated-run analysis, horizon sensitivity, and explicit computational/energy characterization. FT-64 achieves a macro AUC-ROC of \textbf{TBD} and macro recall of \textbf{TBD} across unseen nodes at $H=15$, while requiring approximately 7.52 MMAC and 431 KiB of FP32 weights. On Raspberry Pi 5, batch-1 inference requires \textbf{TBD} ms and \textbf{TBD} mJ, supporting gateway-edge execution. The horizon study further shows \textbf{TBD concise measured trade-off}. Future work will target larger multi-site outdoor campaigns, online drift handling, communication-aware missing-data models, and quantized microcontroller implementation.
```
