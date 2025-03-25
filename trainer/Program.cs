using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System.Drawing;
using System.Drawing.Imaging;
using Microsoft.ML.Vision;

namespace AnimalRecognitionTraining
{
    class Program
    {
        private static readonly string BaseImagesFolder = "images";
        private static readonly string TrainTagsTsv = Path.Combine(BaseImagesFolder, "tags.tsv");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 1);

            // 1. Carica i dati
            IEnumerable<ModelInput> trainingData = LoadImagesFromDirectory(mlContext, BaseImagesFolder, TrainTagsTsv);
            IDataView trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);

            // 2. Definisci la pipeline di trasformazione
            var dataProcessPipeline = mlContext.Transforms
                .Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "Label")
                .Append(mlContext.Transforms.CopyColumns(outputColumnName: "Image", inputColumnName: "Image")); // Copia la colonna "Image"

            // 3. Definisci la pipeline di training
            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                labelColumnName: "Label", featureColumnName: "Image");
            trainer.WithOnFitDelegate((context) =>
            {
                Console.WriteLine($"Addestramento completato");
                Console.WriteLine($"FeatureColumnName: {context.FeatureColumnName}");
            });
            var trainingPipeline = dataProcessPipeline.Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

            // 4. Addestra il modello
            Console.WriteLine("======== Inizia l'addestramento del modello ========");
            ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);
            Console.WriteLine("======== Modello addestrato ========");

            // 5. Valuta il modello (opzionale)
            IDataView testDataView = mlContext.Data.LoadFromEnumerable(LoadImagesFromDirectory(mlContext, BaseImagesFolder, TrainTagsTsv)); // Usa i dati di training anche per il test per semplicità
            IDataView predictions = trainedModel.Transform(testDataView);
            MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label", predictedLabelColumnName: "PredictedLabel");
            Console.WriteLine($"Accuracy: {metrics.MacroAccuracy:P2}");

            // 6. Salva il modello
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "animal_recognition_model.zip");
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelPath);
            Console.WriteLine($"Modello salvato in: {modelPath}");

            Console.WriteLine("Premi un tasto per terminare...");
            Console.ReadKey();
        }

        public static IEnumerable<ModelInput> LoadImagesFromDirectory(MLContext mlContext, string imageFolder, string tagsFile)
        {
            return File.ReadAllLines(tagsFile)
                .Select(line => line.Split('\t'))
                .Select(parts =>
                {
                    string imagePath = Path.Combine(imageFolder, parts[0]);
                    string label = parts[1];
                    try
                    {
                        Bitmap bitmap = new Bitmap(imagePath);

                        // **RIDIMENSIONA L'IMMAGINE**
                        Bitmap resizedBitmap = ResizeImage(bitmap, 224, 224);

                        // Converte l'immagine in un array di float (normalizzato tra 0 e 1)
                        var pixels = resizedBitmap.LockBits(new Rectangle(0, 0, resizedBitmap.Width, resizedBitmap.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                        var bytes = new byte[pixels.Height * pixels.Width * 3];
                        System.Runtime.InteropServices.Marshal.Copy(pixels.Scan0, bytes, 0, bytes.Length);
                        resizedBitmap.UnlockBits(pixels);

                        var floatValues = bytes.Select(b => (float)b / 255.0f).ToArray();

                        return new ModelInput
                        {
                            Image = floatValues,
                            Label = label
                        };
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Errore caricamento immagine {imagePath}: {ex.Message}");
                        return null;
                    }
                })
                .Where(item => item != null)
                .ToList();
        }

        // Funzione per ridimensionare l'immagine
        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(System.Drawing.Drawing2D.WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }
    }

    // Definisci la classe ModelInput
    public class ModelInput
    {
        [ColumnName("Image")]
        [VectorType(224, 224, 3)]
        public float[] Image { get; set; }

        [ColumnName("Label")]
        public string Label { get; set; }
    }

    // Definisci la classe ModelOutput
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }

        [ColumnName("Score")]
        public float[] Score { get; set; }
    }
}