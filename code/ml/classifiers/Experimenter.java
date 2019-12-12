package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.data.*;

public class Experimenter {
	
	public static void main(String[] args) {
//		testData();
//		testBagger();
//		testPerceptronBagging();
//		testTreeBagging();
//		tunePerceptron();
//		tuneBaggedPerceptron();
//		tuneDecisionTree();
//		tuneBaggedDecisionTree();
		compareClassifiers();
	}
	
	public static void testBagger() {
		BaggingClassifier bag = new BaggingClassifier();
		bag.printClassifierConstructor();
	}
	
	public static void testData() {
		DataSet data = new DataSet("data/ionosphere.data", "ionosphere");
		System.out.println(data.getData().size());
		Example e1 = data.getData().get(0);
		System.out.println(e1.toString());
		
	}
	
	public static void tunePerceptron() {
		DataSet data = new DataSet("data/abalone.data", "abalone");
		
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
		DataSet data = new DataSet("data/abalone.data", "abalone");
		
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
		DataSet data = new DataSet("data/abalone.data", "abalone");
		
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
//		DataSet data = new DataSet("data/abalone.data", "abalone");
//		DataSet data = new DataSet("data/ionosphere.data", "ionosphere");
		DataSet data = new DataSet("data/cleveland.data", "cleveland");
		
		CrossValidationSet cvs = new CrossValidationSet(data, 10);
		BaggingClassifier c = new BaggingClassifier();

		for (int n = 1; n <= 20; n++) {
			c.setNumClassifiers(n);
			
			for (int d = 1; d <= 15; d++) {
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
//		DataSet data = new DataSet("data/abalone.data", "abalone");
		DataSet data = new DataSet("data/ionosphere.data", "ionosphere");
		
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
//		DataSet data = new DataSet("data/abalone.data", "abalone");
//		DataSet data = new DataSet("data/ionosphere.data", "ionosphere");
		DataSet data = new DataSet("data/cleveland.data", "cleveland");
		
		DataSetSplit splitData = data.split(0.8);
		BinaryDecisionTreeClassifier tree = new BinaryDecisionTreeClassifier();
		tree.setDepthLimit(1);
		tree.train(splitData.getTrain());
		System.out.println(getAccuracy(tree,splitData.getTest()));
		System.out.println(tree);
		
		BaggingClassifier bag = new BaggingClassifier(10);
		bag.setClassifierConstructor("10t5");
		bag.train(splitData.getTrain());
		System.out.println(getAccuracy(bag,splitData.getTest()));
		
//		bag.printClassifiers();
	}
	
	public static void compareClassifiers() {
		
		DataSet data = new DataSet("data/abalone.data", "abalone");
		
		CrossValidationSet cvs = new CrossValidationSet(data, 10);
		
		Classifier[] classifiers = new Classifier[2];
		
//		PerceptronClassifier p = new PerceptronClassifier();
//		p.setIterations(3);
//		classifiers[0] = p;
//		
//		BaggingClassifier bp = new BaggingClassifier();
//		bp.setNumClassifiers(14);
//		bp.setClassifierConstructor("14p3");
//		classifiers[1] = bp;
		
		BinaryDecisionTreeClassifier tree1 = new BinaryDecisionTreeClassifier();
		tree1.setDepthLimit(7);
		classifiers[0] = tree1;
		
		BaggingClassifier bt1 = new BaggingClassifier();
		bt1.setNumClassifiers(6);
		bt1.setClassifierConstructor("6p7");
		classifiers[1] = bt1;
		
//		BinaryDecisionTreeClassifier tree2 = new BinaryDecisionTreeClassifier();
//		tree1.setDepthLimit(2);
//		classifiers[4] = tree2;
//		
//		BaggingClassifier bt2 = new BaggingClassifier();
//		bp.setNumClassifiers(6);
//		bp.setClassifierConstructor("6p2");
//		classifiers[5] = bt2;

		for (Classifier c : classifiers) {
			for (int i = 0; i < 10; i++) {
				DataSetSplit splitData = cvs.getValidationSet(i);
				c.train(splitData.getTrain());
				System.out.println(getAccuracy(c,splitData.getTest()));
			}
			System.out.println();
		}
		
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
