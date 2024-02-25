using Microsoft.ML.Data;

namespace SentimentPrediction.DataModels;

public class SentimentData
{
    [LoadColumn(0)] 
    public string Text;
    
    [LoadColumn(1), ColumnName("Label")]
    public bool Sentiment;
}