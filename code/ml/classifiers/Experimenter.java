package ml.classifiers;

import java.util.HashMap;
import java.util.Map;

import ml.data.*;

public class Experimenter {
	
	public static void main(String[] args) {
		testData();
		testBagger();
	}
	
	public static void testBagger() {
		BaggingClassifier bag = new BaggingClassifier();
		bag.printConstructor();
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

}
