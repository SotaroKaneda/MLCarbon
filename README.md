# MLCarbon
 End-to-end carbon footprint modeling tool

To test the model using emperical data from Megatron LM and "Carbon Emissions"
```
python3 test/tpu_test.py
python3 test/gpu_test.py
python3 test/mode_test.py
```
\begin{table}[t!]
\centering
\caption{The validation on the operational carbon footprints of various LLMs.}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{|c|c|c|c|}
\hline
LLM             & T5       & GPT3       & XLM  \\\hline\hline
reference       & \multicolumn{2}{c|}{\cite{Patterson:Carbon2021}} & \cite{Wu:MLS2022}\\\hline
developer       & Google   & OpenAI     & Meta \\ \hline
param. \# (B)   & 11       & 175        & 0.55 \\ \hline
token \# (B)    & 500      & 300        & 6K   \\ \hline
$\mathit{CO}_2\mathit{e/KWh}$ & 0.545   & 0.429    & 0.413  \\ \hline
PUE             & 1.12     & 1.1        & 1.1   \\ \hline
device          & TPUv3    & V100       & V100  \\ \hline
device TDP (W)  & 450      & 300        & 300   \\ \hline
avg. power (W)  & 310      & 330        & 342   \\ \hline
peak TFLOPs/s   & 123      & 125        & 125   \\ \hline
actual TFLOPs/s & 45.6     & 24.6       & 26.5     \\ \hline
hardware eff.   & 37\%     & 19.7\%     & 21.2\%   \\ \hline
chip \#         & 512      & 10K        & 512   \\ \hline\hline
total FLOPs     & 4.05E+22 & 3.14E+23   & 2.39+22   \\ \hline
\textbf{predicted FLOPs}   & 3.8E+22 & 3.6E+23      & 2.28E+22   \\ \hline\hline
training days    & 20 & 14.8   & 20.4   \\ \hline
\textbf{predicted days}   & 18.8  & 17.1      & 19.44 \\ \hline\hline

$\mathit{tCO}_2\mathit{e}$ & 46.7  & 552.1     & 39  \\ \hline
\textbf{predicted} $\mathbf{\mathit{tCO}_2\mathit{e}}$ & 43.9 & 638.8 & 37.18   \\ \hline\hline
$\mathbf{\Delta}$ & -6.06\% & +15.7\%     & -4.67\% \\
\hline
\end{tabular}
\label{t:ml_validate_result}
\vspace{-0.1in}
\end{table}
