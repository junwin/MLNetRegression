# Predicting a house price using ML .NET

# House price sample
To understand how the functionality fits into the typical workflow of data preparation, training the model and evaluating the fit using test data sets and using the model. I took a look at implementing a simple regression application to predict the sale price of a house given a simple set of features over about 800 home sales. In the sample, we are going to take a look at a supervised learning problem of Multivariate linear regression. A supervised learning task is used to predice the value of a label from a set of features. In this case, we want to use one or more features to predict the sale price(the label) of a house. 

The focus was on getting a small sample up and running, that can then be used to experiment with the choice of feature and training algorithms. You can find the code for this article on [GitHub here](https://github.com/junwin/MLNetRegression)

We will train the model using a set of sales data to predict the sale price of a house given a set of features over about 800 home sales. While the sample data has a wide range of features, a key aspect of developing a useful system would be to understand the choice of features used affects the model.

# Before you start

You can find a [10 minute tutorial](https://dotnet.microsoft.com/learn/machinelearning-ai/ml-dotnet-get-started-tutorial/intro) that will take you through installing and pre requisits - or just use the following steps.

You will need to have Visual Studio 16.6 or later with the .NET Core installed you can get there [here](https://visualstudio.microsoft.com/downloads/?utm_medium=microsoft&utm_source=docs.microsoft.com&utm_campaign=button+cta&utm_content=download+vs2017)

To start building .NET apps you just need to download and install the [.NET SDK](https://download.visualstudio.microsoft.com/download/pr/48d03227-4429-4daa-ab6a-78840bc61ea5/b815b04bd689bf15a9d95668624a77fb/dotnet-sdk-2.2.104-win-gs-x64.exe).

Install the ML.Net Package by running the following in a command prompt
```
dotnet add package Microsoft.ML --version 0.10.0
```
