using Microsoft.ML;
using SentimentPrediction.DataModels;
using Spectre.Console;
using Console = Spectre.Console.AnsiConsole;

namespace SentimentPrediction;

class Program
{
    static void Main(string[] args)
    {
        var ctx = new MLContext();
        
        // Load data
        var dataView = ctx.Data.LoadFromTextFile<SentimentData>("Resources/yelp_labelled.txt");

        // It is recommended that individual sets for training, testing, and validation are use used when training actual models.
        // As this is just to start grasping the fundamentals, it should be fine.
        // Split data into testing set
        var splitDataView = ctx.Data.TrainTestSplit(dataView, 0.1);
        
        // Build model
        var estimator = ctx.Transforms.Text
            .FeaturizeText("Features", nameof(SentimentData.Text))
            .Append(ctx.BinaryClassification.Trainers.SdcaLogisticRegression(featureColumnName: "Features"));
        
        // Train model
        ITransformer model = default!;

        var rule = new Rule("Create and Train Model");
        Console
            .Live(rule)
            .Start(console =>
            {
                console.Refresh();
                
                model = estimator.Fit(splitDataView.TrainSet);
                var predictions = model.Transform(splitDataView.TestSet);

                rule.Title = "Training Complete, Evaluating Accuracy.";
                console.Refresh();
                
                // evaluate the accuracy of the model
                var metrics = ctx.BinaryClassification.Evaluate(predictions);

                var table = new Table()
                    .MinimalBorder()
                    .Title("Model Accuracy");
                table.AddColumns("Accuracy", "Auc", "F1Score");
                // ROC curve (receiver operating characteristic curve) shows the performance of a classification model at all classification thresholds
                // AUC aggregates measure of performance across all possible classification thresholds
                // F1 score measures the model's balance between precision and recall
                table.AddRow(metrics.Accuracy.ToString("P2"), metrics.AreaUnderRocCurve.ToString("P2"), metrics.F1Score.ToString("P2"));

                console.UpdateTarget(table);
                console.Refresh();
            });

        var engine = ctx.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        while (true)
        {
            var text = AnsiConsole.Ask<string>("Whats your [green]review text[/]? Press [red]q to quit[/].");

            if (text.Equals("q", StringComparison.OrdinalIgnoreCase))
                break;
            
            var input = new SentimentData()
            {
                Text = text,
            };
            var result = engine.Predict(input);
            var resultPrompt = result.Prediction
                ? (color: "green", emoji: "👍")
                : (color: "red", emoji: "👎");
            Console.MarkupLine($"{resultPrompt.emoji} [{resultPrompt.color}]\"{text}\" ({result.Probability:P00})[/] ");

        }

        if (AnsiConsole.Confirm("Would you like to save the model used?"))
        {
            Directory.CreateDirectory("Model");
            ctx.Model.Save(model, dataView.Schema, "Model/model.zip");
        }
    }
}