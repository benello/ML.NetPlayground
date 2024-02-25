using Microsoft.ML.Data;
using SentimentPrediction.DataModels;

namespace SentimentPrediction;

public class SentimentPrediction
    : SentimentData
{
    [ColumnName("PredictedLabel")] 
    public bool Prediction;
    
    public float Probability { get; set; }
    public float Score { get; set; }
}