package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

public class BaggingClassifier implements Classifier {
	private int n;
	private Classifier[] classifiers;
	private DataSet[] splits;
	
	private String constructor;
	// Example: 3p3 2t3
	
	private static Random rand;
	
	public static final char PERCEPTRON = 'p';
	public static final char DECISION_TREE = 't';
	
	/**
	 * Default constructor with n=3
	 */
	public BaggingClassifier() {
		this(3);
	}
	
	/**
	 * Constructor with n classifiers
	 * 
	 * @param numClassifiers
	 */
	public BaggingClassifier(int numClassifiers) {
		n = numClassifiers;
		classifiers = new Classifier[n];
		splits = new DataSet[n];
		
		constructor = new String(new char[n]).replace("\0", "p ");
		constructor = constructor.substring(0, constructor.length()-1);
		
		rand = new Random();
		
	}
	
	public void setConstructor(String s) {
		constructor = s;
	}
	
	
	public void printConstructor() {
		System.out.println(constructor);
	}
	
	/**
	 * train creates n samples of the data and n classifiers
	 * and trains each one on its respective dataset
	 * 
	 * @param DataSet data
	 */
	@Override
	public void train(DataSet data) {
		setupData(data);
		setupClassifiers();
		for (int i = 0; i < n; i++) {
			classifiers[i].train(splits[i]);
		}
	}
	
	/**
	 * 
	 */
	public void setupData(DataSet data) {
		for (int classifier = 0; classifier < n; classifier++) {
			
			DataSet sample = new DataSet(data.getFeatureMap());
			List<Example> currentData = data.getData();
			int size = currentData.size();
			
			for (int i = 0; i < size; i++) 
				sample.addData(currentData.get(rand.nextInt(size)));
			
			splits[classifier] = sample;
		}
	}
	
	/**
	 * 
	 * @param data
	 * @return
	 */
	public void setupClassifiers() {
		
		classifiers = new Classifier[n];
		int i = 0;
		
		String[] batch = constructor.split(" ");
		for (String b : batch) {
			
			int iterations = 1;
			if (Character.isDigit(b.charAt(0))) {
				iterations = b.charAt(0) - '0';
				b.substring(1);
			}
			
			char classifierType = b.charAt(0);
			for (int iter = 0; iter<iterations; iter++) {
				if (i >= n) break;
				
				if (classifierType == PERCEPTRON) {
					PerceptronClassifier c = new PerceptronClassifier();
					// tune hyperparameters if they exist
					if (b.length() > 2) c.setIterations(Integer.valueOf(b.substring(1)));
					classifiers[i] = c; i++;
					
				} else if (classifierType == DECISION_TREE) {
					
					DecisionTreeClassifier c = new DecisionTreeClassifier();
					// tune hyperparameters if they exist
					if (b.length() > 2) c.setDepthLimit(Integer.valueOf(b.substring(1)));
					classifiers[i] = c; i++;
					
				}
			}
		}
	}

	/**
	 * classify has each trained classifier vote for a label
	 * and picks the label with the max vote
	 * 
	 * @param input Example
	 * @return label
	 */
	@Override
	public double classify(Example example) {
		Map<Double, Integer> labelVotes = new HashMap<>();
		for (Classifier c : classifiers) {
			double classification = c.classify(example);
			labelVotes.put(classification, labelVotes.getOrDefault(classification, 0) + 1);
		}
		
		int maxSoFar = -1;
		double bestLabel = 0.0;
		for (double c : labelVotes.keySet()) {
			if (labelVotes.get(c) > maxSoFar) {
				maxSoFar = labelVotes.get(c);
				bestLabel = c;
			}
		}
		
		return bestLabel;
	}

	@Override
	public double confidence(Example example) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	public void printClassifiers() {
		for (Classifier c : classifiers) {
			System.out.println(c);
		}
	}
	
	
	public void printDataSets() {
		for (int i = 0; i < splits[0].getData().size(); i++) {

		}
		
		
	}
	
	/**
	 * 
	 * 
	 */
	public void setNumClassifiers(int numClassifiers) {
		n = numClassifiers;
		classifiers = new Classifier[n];
		splits = new DataSet[n];
	}
}
