using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using System.IO;

namespace myApp
{
    internal class Helpers
    {
        public static void PrintRegressionMetrics(string algorithmName, RegressionMetrics metrics)
        {
            var L1 = metrics.L1;
            var L2 = metrics.L2;
            var RMS = metrics.L1;
            var lossFunction = metrics.LossFn;
            var R2 = metrics.RSquared;

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {algorithmName} Regression model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       L1 Loss:    {L1:0.###} ");
            Console.WriteLine($"*       L2 Loss:    {L2:0.###}  ");
            Console.WriteLine($"*       RMS:          {RMS:0.###}  ");
            Console.WriteLine($"*       Loss Function: {lossFunction:0.###}  ");
            Console.WriteLine($"*       R-squared: {R2:0.###}  ");
            Console.WriteLine($"*************************************************************************************************************");
        }
        /*
                public static void PrintRegressionFoldsAverageMetrics(string algorithmName,
                                                                      (RegressionMetrics metrics,
                                                                       ITransformer model,
                                                                       IDataView scoredTestData)[] crossValidationResults
                                                                     )
                                                                      public static void PrintRegressionFoldsAverageMetrics(string algorithmName, TrainCatalogBase.CrossValidationResult<RegressionMetrics>[] crossValidationResults)
                                                                     */
        public static void PrintRegressionFoldsAverageMetrics(string algorithmName,
                                                           (RegressionMetrics metrics,
                                                            ITransformer model,
                                                            IDataView scoredTestData)[] crossValidationResults
                                                          )
        {
            var L1 = crossValidationResults.Select(r => r.metrics.L1);
            var L2 = crossValidationResults.Select(r => r.metrics.L2);
            var RMS = crossValidationResults.Select(r => r.metrics.L1);
            var lossFunction = crossValidationResults.Select(r => r.metrics.LossFn);
            var R2 = crossValidationResults.Select(r => r.metrics.RSquared);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {algorithmName} Regression model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average L1 Loss:    {L1.Average():0.###} ");
            //Console.WriteLine($"*       Average L2 Loss:    {L2.Average():0.###}  ");
            //Console.WriteLine($"*       Average RMS:          {RMS.Average():0.###}  ");
            //Console.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            Console.WriteLine($"*       Average R-squared: {R2.Average():0.###}  ");
            Console.WriteLine($"*************************************************************************************************************");
        }
    }
}