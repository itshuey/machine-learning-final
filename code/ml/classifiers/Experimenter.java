package ml.classifiers;

import java.util.HashMap;
import java.util.Map;

import ml.data.*;

public class Experimenter {
	
	public static void main(String[] args) {
//		testBagger();
//		testPerceptronBagging();
//		tunePerceptron();
		testTreeBagging();
	}
	
	public static void testBagger() {
		BaggingClassifier bag = new BaggingClassifier();
		bag.printClassifierConstructor();
	}
	
	public static void tunePerceptron() {
		DataSet data = new DataSet("data/abalone.data");
		
		DataSetSplit splitData = data.split(0.8);
		PerceptronClassifier c = new PerceptronClassifier();

		for (int i = 1; i < 20; i++) {
			c.setIterations(i);
			c.train(splitData.getTrain());
			System.out.println(getAccuracy(c,splitData.getTest()));
		}
	}
	
	public static void testPerceptronBagging() {
		DataSet data = new DataSet("data/abalone.data");
		
		DataSetSplit splitData = data.split(0.8);
		PerceptronClassifier c = new PerceptronClassifier();
		c.train(splitData.getTrain());
		System.out.println(getAccuracy(c,splitData.getTest()));
//		System.out.println(c);
		
//		TwoLayerNN net = new TwoLayerNN(3);
//		net.train(splitData.getTrain());
//		System.out.println(getAccuracy(net,splitData.getTest()));
		
		BaggingClassifier bag = new BaggingClassifier(10);
		bag.train(splitData.getTrain());
		System.out.println(getAccuracy(bag,splitData.getTest()));
		
//		bag.printClassifiers();
	}
	
	
	public static void testTreeBagging() {
		DataSet data = new DataSet("data/abalone.data");
		
		DataSetSplit splitData = data.split(0.8);
		BinaryDecisionTreeClassifier tree = new BinaryDecisionTreeClassifier();
//		tree.setDepthLimit(3);
		tree.train(splitData.getTrain());
		System.out.println(getAccuracy(tree,splitData.getTest()));
//		System.out.println(tree);
		
		BaggingClassifier bag = new BaggingClassifier(10);
		bag.setClassifierConstructor("10t3");
		bag.train(splitData.getTrain());
		System.out.println(getAccuracy(bag,splitData.getTest()));
		
//		bag.printClassifiers();
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

}
