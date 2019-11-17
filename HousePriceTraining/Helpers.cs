using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;

namespace myApp
{
    internal class Helpers
    {
        public static void PrintRegressionMetrics(string algorithmName, RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {algorithmName} Regression model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       L1 Loss:    {metrics.MeanAbsoluteError:0.###} ");
            Console.WriteLine($"*       L2 Loss:    {metrics.MeanSquaredError:0.###}  ");
            Console.WriteLine($"*       RMS:          {metrics.RootMeanSquaredError:0.###}  ");
            Console.WriteLine($"*       Loss Function: {metrics.LossFunction:0.###}  ");
            Console.WriteLine($"*       R-squared: {metrics.RSquared:0.###}  ");
            Console.WriteLine($"*************************************************************************************************************");
        }

        /*
         * ML NET V11 needs a different signature
          public static void PrintRegressionFoldsAverageMetrics(string algorithmName, TrainCatalogBase.CrossValidationResult<RegressionMetrics>[] crossValidationResults)
         */

        /*
       public static void PrintRegressionFoldsAverageMetrics(string algorithmName,
                                                          (RegressionMetrics metrics,
                                                           ITransformer model,
                                                           IDataView scoredTestData)[] crossValidationResults
                                                         )
                                                         */

        public static void PrintRegressionFoldsAverageMetrics(string algorithmName, IReadOnlyList<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> crossValidationResults)
        {
            Console.WriteLine($"**********************************************");
            Console.WriteLine($"* Metrics for {algorithmName} Regression      ");
            Console.WriteLine($"*---------------------------------------------");

            foreach (var result in crossValidationResults)
            {
                Console.WriteLine($"* R-squared: {result.Metrics.RSquared:0.###}  ");
            }

            Console.WriteLine($"**********************************************");
        }
    }
}