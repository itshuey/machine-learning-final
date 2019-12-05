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
	
	private static Random rand;
	
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
		rand = new Random();
	}
	
	/**
	 * train creates n samples of the data and n classifiers
	 * and trains each one on its respective dataset
	 * 
	 * @param DataSet data
	 */
	@Override
	public void train(DataSet data) {
		setup(data);
		for (int i = 0; i < n; i++) {
			classifiers[i].train(splits[i]);
		}
	}
	
	/**
	 * 
	 */
	public void setup(DataSet data) {
		for (int classifier = 0; classifier < n; classifier++) {
			splits[classifier] = getSampledData(data);
			classifiers[classifier] = getNewClassifier();
		}
	}
	
	/**
	 * 
	 * @param data
	 * @return
	 */
	public static DataSet getSampledData(DataSet data) {
		DataSet sample = new DataSet(data.getFeatureMap());
		List<Example> currentData = data.getData();
		int size = currentData.size();
		
		for (int i = 0; i < size; i++) 
			sample.addData(currentData.get(rand.nextInt(size)));
		
		return sample;
	}
	
	/**
	 * 
	 * @param data
	 * @return
	 */
	public static Classifier getNewClassifier() {
	
//		DecisionTreeClassifier newTree = new DecisionTreeClassifier();
//		newTree.setDepthLimit(3);
//		return newTree;
		
		return new PerceptronClassifier();
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
