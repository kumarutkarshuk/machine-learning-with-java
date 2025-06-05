package app.vercel.utkarshkumar;

import org.tribuo.*;
import org.tribuo.common.tree.AbstractCARTTrainer;
import org.tribuo.common.tree.RandomForestTrainer;
import org.tribuo.data.csv.CSVIterator;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.ensemble.AveragingCombiner;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.rtree.CARTRegressionTrainer;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Paths;

public class WineQualityRegression {
    public static final String DATASET_PATH = "src/main/resources/dataset/winequality-red.csv";
    public static final String MODEL_PATH = "src/main/resources/model/winequality-red-regressor.ser";
    public Model<Regressor> model;
    public Trainer<Regressor> trainer;
    public Dataset<Regressor> trainSet;
    public Dataset<Regressor> testSet;

    void createDatasets() throws Exception {
        RegressionFactory regressionFactory = new RegressionFactory();
        CSVLoader<Regressor> csvLoader = new CSVLoader<>(';', CSVIterator.QUOTE, regressionFactory);
        DataSource<Regressor> dataSource = csvLoader.loadDataSource(Paths.get(DATASET_PATH), "quality");

        // split the dataset into 70% training and 30% test sets using TrainTestSplitter
        TrainTestSplitter<Regressor> dataSplitter = new TrainTestSplitter<>(dataSource, 0.7, 1L);
        trainSet = new MutableDataset<>(dataSplitter.getTrain());
        testSet = new MutableDataset<>(dataSplitter.getTest());
    }

    // train by combining & averaging 10 decision trees (yes-no like questions) with no max depth,
    // min six examples per split (question) and minimizing the mean squared error average error
    // decision trees are logged as model 0, model 1... in the console
    void createTrainer() {
//        Classification and Regression Tree (CART)
        CARTRegressionTrainer subsamplingTree = new CARTRegressionTrainer(
                Integer.MAX_VALUE,
                AbstractCARTTrainer.MIN_EXAMPLES,
                0.001f,
                0.7f,
                new MeanSquaredError(),
                Trainer.DEFAULT_SEED
        );

        trainer = new RandomForestTrainer<>(subsamplingTree, new AveragingCombiner(), 10);
        model = trainer.train(trainSet);
    }

    void evaluate(Model<Regressor> model, String datasetName, Dataset<Regressor> dataset) {
        RegressionEvaluator evaluator = new RegressionEvaluator();
        RegressionEvaluation evaluation = evaluator.evaluate(model, dataset);
        // dimension - the output variable we want to predict
        Regressor dimension0 = new Regressor("DIM-0", Double.NaN); // first dimension (quality in this case)

        // Lower MAE and RMSE values, and higher R^2 value indicate better predictive performance.
        System.out.println("MAE: " + evaluation.mae(dimension0));// MAE (Mean Absolute Error)
        System.out.println("RMSE: " + evaluation.rmse(dimension0)); // RMSE (Root Mean Squared Error)
        // R^2 (Coefficient of Determination)
        // indicates how well the model explains variance in the training and testing data.
        // variance - how much the data varies/is-away from the mean
        System.out.println("R^2: " + evaluation.r2(dimension0));
    }

    void evaluateModels() throws Exception {
        System.out.println("Testing model...");
        evaluate(model, "testSet", testSet);
    }

    // serialize the trained model to a file
    void saveModel() throws Exception {
        File modelFile = new File(MODEL_PATH);
        try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream(modelFile))) {
            objectOutputStream.writeObject(model);
        }
    }

    public static void main(String[] args) throws Exception {
        WineQualityRegression wineQualityRegression = new WineQualityRegression();

        wineQualityRegression.createDatasets();
        wineQualityRegression.createTrainer();
        wineQualityRegression.evaluateModels();
        wineQualityRegression.saveModel();
    }

}
