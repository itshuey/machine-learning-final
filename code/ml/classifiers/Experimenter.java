package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.data.*;

public class Experimenter {
	
	public static void main(String[] args) {
//		testBagger();
//		testPerceptronBagging();
//		tunePerceptron();
//		tuneBaggedPerceptron();
//		tuneDecisionTree();
		tuneBaggedDecisionTree();
//		testTreeBagging();
	}
	
	public static void testBagger() {
		BaggingClassifier bag = new BaggingClassifier();
		bag.printClassifierConstructor();
	}
	
	public static void tunePerceptron() {
		DataSet data = new DataSet("data/abalone.data");
		
		CrossValidationSet cvs = new CrossValidationSet(data, 10);
		PerceptronClassifier c = new PerceptronClassifier();

		for (int it = 1; it <= 30; it++) {
			c.setIterations(it);
			
			double accuracy = 0.0;
			for (int i = 0; i < 10; i++) {
				DataSetSplit splitData = cvs.getValidationSet(i);
				c.train(splitData.getTrain());
				accuracy += getAccuracy(c,splitData.getTest());
			}
			
			System.out.println(accuracy/10);
		}
	}
	
	public static void tuneBaggedPerceptron() {
		DataSet data = new DataSet("data/abalone.data");
		
		CrossValidationSet cvs = new CrossValidationSet(data, 10);
		BaggingClassifier bag = new BaggingClassifier();

		for (int n = 1; n <= 10; n++) {
			int nc = n*5;
			bag.setNumClassifiers(nc);
			bag.setClassifierConstructor(nc+"p3");
			
			double accuracy = 0.0;
			for (int i = 0; i < 10; i++) {
				DataSetSplit current = cvs.getValidationSet(i);
				bag.train(current.getTrain());
				accuracy += getAccuracy(bag,current.getTest());
			}
			
			System.out.println(accuracy/10);
		}
	}
	
	public static void tuneDecisionTree() {
		DataSet data = new DataSet("data/abalone.data");
		
		CrossValidationSet cvs = new CrossValidationSet(data, 10);
		BinaryDecisionTreeClassifier c = new BinaryDecisionTreeClassifier();

		for (int d = 1; d <= 8; d++) {
			c.setDepthLimit(d);
			
			double accuracy = 0.0;
			for (int i = 0; i < 10; i++) {
				DataSetSplit splitData = cvs.getValidationSet(i);
				c.train(splitData.getTrain());
				accuracy += getAccuracy(c,splitData.getTest());
			}
			
			System.out.println(accuracy/10);
		}
	}
	
	public static void tuneBaggedDecisionTree() {
		DataSet data = new DataSet("data/abalone.data");
		
		CrossValidationSet cvs = new CrossValidationSet(data, 10);
		BaggingClassifier c = new BaggingClassifier();

		for (int n = 1; n <= 20; n++) {
			c.setNumClassifiers(n);
			
			for (int d = 1; d <= 8; d++) {
				c.setClassifierConstructor(n + "t" + d);
				
				double accuracy = 0.0;
				for (int i = 0; i < 10; i++) {
					DataSetSplit splitData = cvs.getValidationSet(i);
					c.train(splitData.getTrain());
					accuracy += getAccuracy(c,splitData.getTest());
				}
				System.out.println(accuracy/10);
			}
			
			System.out.println();
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
