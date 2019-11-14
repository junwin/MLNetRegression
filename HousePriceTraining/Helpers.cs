
using Microsoft.ML.Data;
using System;
using System.Linq;

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
        */
    }
}