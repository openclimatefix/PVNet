# Analysis Report: Forecast Evaluation

## Introduction

In this report, we evaluate the performance of OCFs national PV forecast (PVNet) against the Shefield solar PVLive Intraday forecast. The PVLive Updated forecast will be used as the truth for this comparison. This report looks into various error metrics and discuss the implications of our findings.


## Table of Contents
1. [Data Overview](#data-overview)
2. [PVNet Model Evaluation](#pvnet-model-evaluation)
    * [Results Table](#pvnet-results-table)
    * [Plots](#pvnet-plots)
    * [Heatmaps](#pvnet-heatmaps)
    * [Evaluation Across Horizons](#pvnet-horizons)
    * [Probabalistic Forecasts Evaluation](#pvnet-prob)
    * [Ramp Rates Across Horizons](#pvnet-prob)
3. [PVLive Intraday and PVNet vs PVLive Updated](#pvlin-and-pvnet-vs-pvlup)
    * [Results Table](#pvlup)
    * [Plots](#error-metrics)
    * [Heatmaps](#heatmaps)
    * [Real Time Ramp Rates](#real-time-ramp-rates)
    * [Error Distribution](#error-distribution)
4. [Relative Performance of PVNet vs PVLive](#pvnet-vs-pvlive)
5. [Conclusion](#conclusion)
6. [Appendix](#mathematical-formulas)

---

## Data Overview

The aim of these results which to evaluate the performance of the OCF PVNet model against the PVLive Intraday forecast over 2022. However, we are slightly restricted by the amount of data that is available.

OCF only starting recording the data from PVLive intraday from 2022-06-14.

The OCF PVNet is also missing forecasts that would not be representive of how our forecast would perform in production, due to us not having certain bits of data avaiable when carrying out this backtest of our latest model. These are.

- ~1 month of missing NWP data
- ~1 month of missing satelitte data
- 2 days every month in which the statelittes switch over.

Hence, this report is broken into different sections to fairly evaluate each PVNet and PVLive Intraday against each other. Firstly Just PVNet is evaluated against PVLive Updated. Then, for the data that is available for both PVLive Intraday and PVNet, a further similar evaluation process is carried out.




- **Root Mean Square Error (RMSE)**: Indicates the sample standard deviation of the differences between predicted and observed values.

---



## PVNet Model Evaluation

PVNet was evaluated against PVLive updated for the period of:
* 2022-01-01 to 2022-11-19

The initial evaluation only consideres the 0th horizon forefats (aka the realtime forecast). Evaluation across different horizons can be found later on in this section. 

### PVNet Results Table

| Metric | PVNet ± Standard Erorr| Standard Deviation
|--------|-----------------------|-------------------
| MAE    | 126.92 ± 1.80         |200.40
| RMSE   | 237.20 ± 3.52         |392.69            
| MBE    | -11.01 ± 2.12         |237.00            
| R^2    | 0.990                 |
| RTRR   | 96.40 ± 1.36          |152.00

* RTRR: Real Time Ramp Rate (See formula in Apendix)

### PVNet Plots

![PVNet MAE](./imgs/pvnet_all_MAE_monthly.png)
*Fig 1: Average monthly MAE for PVNet across 2022.*

![PVNet RMSE](./imgs/pvnet_all_RMSE_monthly.png)
*Fig 2: Average monthly RMSE for PVNet across 2022.*

![PVNet MBE](./imgs/pvnet_all_MBE_monthly.png)
*Fig 3: Average monthly MBE for PVNet across 2022.*





### PVNet Heatmaps

<!-- ![Error Distribution of Model A](./pvnet_comp_imgs/full_pvnet/coolwarm_v2/pvnet_all_heatmap_MAE_month.png)
*Figure 2: Distribution of forecast errors for Model A.*

![Error Distribution of Model A](./pvnet_comp_imgs/full_pvnet/coolwarm_v2/pvnet_all_heatmap_MAE_week.png)
*Figure 2: Distribution of forecast errors for Model A.* -->


<table>
<tr>
<td><img src="./imgs/pvnet_all_heatmap_MAE_month.png"/></td>
<td><img src="./imgs/pvnet_all_heatmap_MAE_week.png"/></td>
</tr>
<tr>
<td>Fig 4: PVNet MAE heatmap for Hour vs Month.</td>
<td>Fig 5: PVNet MAE heatmap for Hour vs Week.</td>
</tr>
</table>



<table>
<tr>
<td><img src="./imgs/pvnet_all_heatmap_MBE_month.png"/></td>
<td><img src="./imgs/pvnet_all_heatmap_MBE_week.png"/></td>
</tr>
<tr>
<td>Fig 6: PVNet MBE heatmap for Hour vs Month.</td>
<td>Fig 7: PVNet MBE heatmap for Hour vs Week.</td>
</tr>
</table>





<!-- ![Model Forecast vs. Actuals](./pvnet_comp_imgs/full_pvnet/coolwarm_v2/pvnet_all_heatmap_MBE_month.png)
*Figure 1: Comparison of model forecast vs. actuals.*

![Model Forecast vs. Actuals](./pvnet_comp_imgs/full_pvnet/coolwarm_v2/pvnet_all_heatmap_MBE_week.png)
*Figure 1: Comparison of model forecast vs. actuals.* -->



### Evaluation Across Horizons


![PVNet Horizon MAE](./imgs/pvnet_all_horizon_MAE.png)
*Fig 8: PVNet horizon vs MAE averaged across 2022.*

![PVNet Horizon RMSE](./imgs/pvnet_all_horizon_RMSE.png)
*Fig 10: PVNet horizon vs RMSE averaged across 2022.*

![PVNet Horizon MBE](./imgs/pvnet_all_horizon_MBE.png)
*Fig 11: PVNet horizon vs MBE averaged across 2022.*



### Probabalistic Forecasts Evaluation

Pinball Scores

% of predicitons over certain value



## PVlin and PVNet vs PVlup

### Results Table

| Forecast | MAE                     | RMSE                   | R2     | MBE                     |RTRR
|----------|-------------------------|------------------------|--------|-------------------------|----
| PVLin    | 197.37 ± 4.19           | 357.56 ± 7.33          | 0.978  | 187.91 ± 4.28           |64.96 ± 1.47
| PVNet    | 129.75 ± 2.73           | 233.49 ± 5.21          | 0.99   | 6.30 ± 3.28             |100.69 ± 2.17

The ± represents the standard error in the metric.

### Standard Deviations of Metrics

| Forecast | Std MAE        | StdRMSE       | Std MBE     |Std RTRR
|----------|----------------|---------------|-------------|---------
| PVLin    | 298.17         | 521.04        | 304.23      |104.73
| PVNet    | 194.14         | 370.62        | 233.43      |152.01



### Plots

![Model Forecast vs. Actuals](./imgs/pvnet_vs_pvlin_MAE_monthly.png)
*Fig : Average monthly MAE for PVNet and PVLive Intraday*

![Error Distribution of Model A](./imgs/pvnet_vs_pvlin_RMSE_monthly.png)
*Fig : Average monthly RMSE for PVNet and PVLive Intraday*

![Error Distribution of Model A](./imgs/pvnet_vs_pvlin_MBE_monthly.png)
*Fig : Average monthly MBE for PVNet and PVLive Intraday*

### Heatmap


MAE

<table>
<tr>
<td><img src="./imgs/pvlin_all_heatmap_MAE_Month.png"/></td>
<td><img src="./imgs/pvnet_all_heatmap_MAE_Month.png"/></td>
</tr>
<tr>
<td>Fig : PVLive Intraday MAE heatmap for Hour vs Month.</td>
<td>Fig : PVNet MAE heatmap for Hour vs Month.</td>
</tr>
</table>

<table>
<tr>
<td><img src="./imgs/pvlin_all_heatmap_MAE_Week.png"/></td>
<td><img src="./imgs/pvnet_all_heatmap_MAE_Week.png"/></td>
</tr>
<tr>
<td>Fig : PVLive Intraday MAE heatmap for Hour vs Week.</td>
<td>Fig : PVNet MAE heatmap for Hour vs Week.</td>
</tr>
</table>



MBE


<table>
<tr>
<td><img src="./pvnet_comp_imgs/pvnet_vs_pvlin/v2/plots/heatmaps/pvlin_all_heatmap_MBE_Month.png"/></td>
<td><img src="./pvnet_comp_imgs/pvnet_vs_pvlin/v2/plots/heatmaps/pvnet_all_heatmap_MBE_Month.png"/></td>
</tr>
<tr>
<td>Fig : PVLive Intraday MBE heatmap for Hour vs Month.</td>
<td>Fig : PVNet MBE heatmap for Hour vs Month.</td>
</tr>
</table>


<table>
<tr>
<td><img src="./pvnet_comp_imgs/pvnet_vs_pvlin/v2/plots/heatmaps/pvlin_all_heatmap_MBE_week.png"/></td>
<td><img src="./pvnet_comp_imgs/pvnet_vs_pvlin/v2/plots/heatmaps/pvnet_all_heatmap_MBE_week.png"/></td>
</tr>
<tr>
<td>Fig : PVLive Intraday MBE heatmap for Hour vs Week.</td>
<td>Fig : PVNet MBE heatmap for Hour vs Week.</td>
</tr>
</table>

### Error Distribution

![Model Forecast vs. Actuals](./imgs/pvnet_pvlin_Error_nights_excluded_kde.png)
*Fig : Kernel density estimation of error distribution of PVNet and PVLive Inraday, with nights removed.*

![Model Forecast vs. Actuals](./imgs/pvnet_pvlin_Error_nights_excluded_hist.png)
*Fig : Histrogram of error distribution of PVNet and PVLive Inraday, with nights removed.*



## Relative Performance of PVNet vs PVLive





---

## Conclusion

From this analysis

---



## Appendix

### Error Metrics
To compute the error metrics, we utilised the following mathematical formulas:

1. **MAE (Mean Absolute Error)**:
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_{\text{true},i} - y_{\text{pred},i} \right|$$

- **Mean Absolute Error (MAE)**: Represents the average of the absolute differences between the forecasted and actual values. It provides an idea of the magnitude of errors.

2. **RMSE (Root Mean Squared Error)**:
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( y_{\text{true},i} - y_{\text{pred},i} \right)^2}$$

3. **R^2 (Coefficient of Determination)**:
$$R^2 = 1 - \frac{\sum_{i=1}^{n} \left( y_{\text{true},i} - y_{\text{pred},i} \right)^2}{\sum_{i=1}^{n} \left( y_{\text{true},i} - \bar{y_{\text{true}}} \right)^2}$$

Where $\bar{y_{\text{true}}}$ is the mean of the true values.

4. **MBE (Mean Bias Error)**:
$$\text{MBE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_{\text{pred},i} - y_{\text{true},i} \right)$$

5. **RealTimeRampRateDiff**:
$$\text{RampRateDiff} = \frac{1}{n} \sum_{i=1}^{n} \left| \Delta y_{\text{true},i} - \Delta y_{\text{pred},i} \right|$$
Where $\Delta y$ represents the difference between consecutive values.

<!-- 6. **Pinball Loss**:
$$\text{Pinball Loss} = \frac{1}{n} \sum_{i=1}^{n} \left( y_{\text{true},i} - y_{\text{pred},i} \right) \times (\tau - \mathbb{1}(y_{\text{true},i} < y_{\text{pred},i}))$$
Where \(\tau\) is a quantile (0.5 in your case) and \(\mathbb{1}(.)\) is the indicator function. -->

### Variability of Error Metrics

The **Standard Deviations** of these metrics are calculated as follows:
$$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \text{Metric}_i - \bar{\text{Metric}} \right)^2}$$
Where $\bar{\text{Metric}}$ is the mean of the metric across all data points.

The **Standard Error of The Mean** of these metrics are calcuated as follows:
$$ SE = \frac{\sigma}{\sqrt{n}}$$



