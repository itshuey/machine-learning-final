package ml.classifiers;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

/**
 * BaggingClassifier 
 * 
 * @author huey
 */
public class BaggingClassifier implements Classifier {
	private int n;
	private Classifier[] classifiers;
	private DataSet[] splits;
	
	private String constructor;
	// Separate the instructions by a space
	// Each instruction is of form $n$c$h
	// 	where $n = number of classifiers (optional)
	//		  $c = character code of classifier
	//		  $h = hyperparameter to tune (optional)
	//
	// Example: 3p t
	// Example: p3 4t3

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
		
		// Default: create n perceptrons
		constructor = n + "p";
		rand = new Random();
	}
	
	/**
	 * train creates n samples of the data and n classifiers
	 * and trains each one on its respective dataset
	 * 
	 * @param DataSet data
	 */
	public void train(DataSet data) {
		train(data, true);
	}
	
	/**
	 * train creates n samples of the data and n classifiers
	 * and trains each one on its respective dataset
	 * 
	 * @param DataSet data
	 * @param boolean reset: whether to reset the classifiers and data
	 */
	public void train(DataSet data, boolean reset) {
		
		if (reset) {
			setupData(data);
			setupClassifiers();
		}
		
		for (int i = 0; i < n; i++) {
			classifiers[i].train(splits[i]);
		}
	}
	
	/**
	 * setUpClassifiers creates and sets the hyperparameters
	 * based on the classifier constructor string
	 * 
	 */
	public void setupClassifiers() {
		
		int i = 0;
		
		// parse each batch of constructors separately
		String[] batch = constructor.split(" ");
		for (String b : batch) {
			
			// find how long the leading number is
			int numEndIndex = 0;
			while(Character.isDigit(b.charAt(numEndIndex))) numEndIndex++;
			
			// save the leading number as the number of iterations (if it exists)
			int iterations = numEndIndex == 0 ? 1 : Integer.valueOf(b.substring(0, numEndIndex));
			// get rid of the leading number
			b = b.substring(numEndIndex);
			
			// parse the classifier type
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
	 * setUpData creates n new datasets by
	 * sampling from the input dataset randomly with replacement 
	 * 
	 * @param input DataSet
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
	 * classify returns the label with the max vote
	 * from the trained classifiers
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

	/**
	 * confidence returns the average confidence of the
	 * label with the max vote
	 * 
	 * @param input Example
	 * @return confidence
	 */
	@Override
	public double confidence(Example example) {
		double classification = classify(example);
		double confidence = 0.0;
		int numVotes = 0;
		
		for (Classifier c : classifiers) {
			if (c.classify(example) == classification) {
				confidence += c.confidence(example);
				numVotes++;
			}
		}
		
		return confidence / numVotes;
	}
	
	///////////////
	//  UTILITY
	///////////////
	
	/**
	 * Prints each of the classifiers
	 */
	public void printClassifiers() {
		for (Classifier c : classifiers) {
			System.out.println(c);
		}
	}
	
	/**
	 * Prints the classifier constructor string
	 */
	public void printClassifierConstructor() {
		System.out.println(constructor);
	}
	
	/**
	 * Prints the data sets
	 */
	public void printDataSets() {
		for (int i = 0; i < splits[0].getData().size(); i++) {

		}
		
		
	}
	
	///////////////////////
	//  GETTERS n SETTERS
	///////////////////////
	
	/**
	 * Sets the number of classifiers
	 * This wipes the existing classifiers and data
	 * 
	 * @param number of classifiers
	 */
	public void setNumClassifiers(int numClassifiers) {
		n = numClassifiers;
		classifiers = new Classifier[n];
		splits = new DataSet[n];
	}
	
	/**
	 * Sets the classifier constructor string
	 * 
	 * @param String classifierConstructor
	 */
	public void setClassifierConstructor(String classifierConstructor) {
		this.constructor = classifierConstructor;
	}
	
	/**
	 * Sets an individual classifier in our array
	 * 
	 * @param int i: classifier to replace
	 * @param Classifier c: new classifier
	 */
	public void setClassifier(int i, Classifier c) {
		classifiers[i] = c;
	}
}
