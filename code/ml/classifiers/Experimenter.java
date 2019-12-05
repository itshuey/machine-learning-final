package ml.classifiers;

import java.util.HashMap;
import java.util.Map;

import ml.data.*;

public class Experimenter {
	
	public static void main(String[] args) {
		testData();
	}
	
	public static void testData() {
		DataSet data = new DataSet("data/abalone.data");
		
		DataSetSplit splitData = data.split(0.8);
		PerceptronClassifier c = new PerceptronClassifier();
		c.train(splitData.getTrain());
		System.out.println(getAccuracy(c,splitData.getTest()));
		System.out.println(c);
		
		BaggingClassifier bag = new BaggingClassifier(10);
		bag.train(splitData.getTrain());
		System.out.println(getAccuracy(bag,splitData.getTest()));
		
		bag.printClassifiers();
	}
	
	
	public static double getAccuracy(Classifier classifier, DataSet testData) {
		int correct = 0;
		int total = 0;
			
		for (Example e : testData.getData()) {
			double classification = classifier.classify(e);
//			System.out.println(classification);
//			System.out.println(e.getLabel());
//			System.out.println();
			if (classification == e.getLabel()) {
				correct++;
			}
			total++;
		}
		
		return (double) correct / total;
	}
	
	public static void experimentThree() {
		DataSet data = new DataSet("data/titanic-train.csv", DataSet.CSVFILE);
		CrossValidationSet cvs = new CrossValidationSet(data, 10);
		double[][] accuracies = new double[10][10];
		
		for (int numHiddenNodes = 1; numHiddenNodes <= 10; numHiddenNodes++) {
			TwoLayerNN net = new TwoLayerNN(numHiddenNodes);
			for (int split = 0; split < 10; split++) {
				DataSetSplit splitData = cvs.getValidationSet(split);
				net.train(splitData.getTrain());
				// add to accuracy list
				accuracies[numHiddenNodes-1][split] =  net.getAccuracy(splitData.getTest());
			}
		}
		
		for (int nodes = 0; nodes < 10; nodes++) {
			System.out.println("Accuracies With " + (nodes+1) + " Hidden Nodes");
			for (double accuracy : accuracies[nodes]) {
				System.out.println(accuracy);
			}
			System.out.println();
		}
	}
	
	public static void experimentFour() {
		DataSet data = new DataSet("data/titanic-train.csv", DataSet.CSVFILE);
		CrossValidationSet cvs = new CrossValidationSet(data, 10);
		Map<Double, double[][]> etaAccuracy = new HashMap<>();

		for (double eta = 0.1; eta <= 1; eta += 0.1) {
			double[][] accuracies = new double[10][10];
			System.out.println("Eta: " + eta);
			
			for (int numHiddenNodes = 1; numHiddenNodes <= 10; numHiddenNodes++) {
				TwoLayerNN net = new TwoLayerNN(numHiddenNodes);
				for (int split = 0; split < 10; split++) {
					DataSetSplit splitData = cvs.getValidationSet(split);
					net.train(splitData.getTrain());
					net.setEta(eta);
					// add to accuracy list
					accuracies[numHiddenNodes-1][split] =  net.getAccuracy(splitData.getTest());
				}
			}
			
			for (int nodes = 0; nodes < 10; nodes++) {
				System.out.println("Accuracies With " + (nodes+1) + " Hidden Nodes");
				for (double accuracy : accuracies[nodes]) {
					System.out.println(accuracy);
				}
				System.out.println();
			}		
		}
	}
}
