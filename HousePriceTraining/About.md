
# Applying ML.NET to a regression problem

ML.NET is an opensource cross-platform machine learning framework intended for .NET developers. It provides a great set of tools to let you implement machine learning applications using .NET – you can find out more about ML.NET [here](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)

To understand how the functionality fits into the typical workflow of accessing data, selecting features, normalisation, training the model and evaluating the fit using test data sets. I took a look at implementing a simple regression application to predict the sale price of a house given a simple set of features over about 800 home sales.

The focus was on getting a small sample up and running, that can then be used to experiment with the choice of feature and training algorithms. You can find the code for this article on GitHub [here](https://github.com/junwin/MLNetRegression)



ML.NET provides a developer friendly API for machine learning, in terms of:
	• Transforms(Feature selection, Text, Schema, Categorical, data normalization, handling missing data)
	• Learners(Linear, Boosted trees, SVM, K-Means)
	• Tools(Data framework, Evaluators, calibrators, Data Loaders)

Put together these support the typical ML workflow:
	• Data preparation(loading and feature extraction)
	• Training (Training and evaluating models)
	• Running( using the trained model)

A big advantage of using the ML.Net framework is that it allows the user to easily experiment with different learning algorithms, changing the set of features, sizes of training and test datasets to get the best results for their problem. This avoids a common issue where teams spend a lot of time collecting unnecessary data and produce models that do not perform well.


Key Concept

When discussing ML.Net it is important to recognize to use of:
	• Transformers - these convert and manipulate data and produce data as an output.
	• Estimators - these take data and produce a transformer or a model, e.g. when training
	• Prediction - this takes a single row of features and predicts a single row of results.  

	We will see how these come into play in the simple regression sample.

ML.NET lets you develop a range of ML systems
	• Forecasting/Regression
	• Issue Classification
	• Predictive maintenance
	• Image classification
	• Sentiment Analysis
	• Recommender Systems
	• Clustering systems

In the sample we are going to take a look a supervised learning problem of 
Multivariate linear regression. In this case we want use one or more features to predict the sale price of a house.

We will train the model using a set of sales data to predict the sale price of a house given a set of features over about 800 home sales.

While the sample data has a wide range of features, a key aspect of developing a useful system would be to understand the choice of features used affects the model.

	1) Training and Saving the model
	
	Our first job is to define a simple data class that we can use when loading our .csv file of house data.
	The important part to note is the [LoadColumn()] attributes, these allow us to associate the  fields to different columns in the input.
	It gives us a simple way to adapt to changes in the data sets we can process.
	The class is also used when we want to predict the price of some house.
	We do not need to use all the fields in the class when training the model.
	
	```c#
	public class HouseData
	    {
	        [LoadColumn(3)]
	        public float Area;
	
	        [LoadColumn(4)]
	        public float Rooms;
	
	        [LoadColumn(13)]
	        public float BedRooms;
	
	        [LoadColumn(12)]
	        public float BedRoomsBsmt;
	
	        [LoadColumn(5)]
	        public float FullBath;
	
	        [LoadColumn(6)]
	        public float HalfBath;
	
	        [LoadColumn(7)]
	        public float Floors;
	
	        [LoadColumn(9)]
	        public float SoldPrice;
	
	        [LoadColumn(22)]
	        public float LotSize;
	
	        [LoadColumn(16)]
	        public string GarageType;
	    }
		```
	
Given we have our data class, we can now read the date from the file.

```c#
          Console.WriteLine("Training product forecasting");

            // Read the sample data into a view that we can use for training
            var trainingDataView = mlContext.Data.ReadFromTextFile<HouseData>(dataPath, hasHeader: true, separatorChar: ',');

            // create the trainer we will use  - ML.NET supports different training methods
            var trainer = mlContext.Regression.Trainers.FastTree(labelColumn: DefaultColumnNames.Label, featureColumn: DefaultColumnNames.Features);

            // Create the training pipeline, this determines how the input data will be transformed
            // We can also select the features we want to use here, the names used correspond to the porperty names in 
            // HouseData
            string[] numericFeatureNames = { "Area","Rooms", "BedRooms", "BedRoomsBsmt", "FullBath", "HalfBath", "Floors","LotSize"};

            // We distinguish between features that are strings e.g. {"attached","detached","none") garage types and 
            // Numeric faeature, since learning systems only work with numeric values we need to convert the strings.
            // You can see that in the training pipeline we have applied OneHotEncoding to do this.
            string[] categoryFeatureNames = { "GarageType" };
            
            var trainingPipeline = mlContext.Transforms.Concatenate(NumFeatures, numericFeatureNames)
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(CatFeatures, inputColumnName: categoryFeatureNames[0]))
                .Append(mlContext.Transforms.Concatenate(DefaultColumnNames.Features, NumFeatures, CatFeatures))
                .Append(mlContext.Transforms.CopyColumns(DefaultColumnNames.Label, inputColumnName: nameof(HouseData.SoldPrice)))
                .Append(trainer);

            // Split the data 90:10 into train and test sets, train and evaluate.
            var (trainData, testData) = mlContext.Regression.TrainTestSplit(trainingDataView, testFraction: 0.2);

			```

Evaluation
