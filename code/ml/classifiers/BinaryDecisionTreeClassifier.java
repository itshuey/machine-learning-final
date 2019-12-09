package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

/**
 * Decision tree classifier that supports real valued features.
 * 
 * The classifier handles real-valued features by splitting based on values greater than
 * and lower than a certain value.
 * 
 * @author dkauchak & huey
 *
 */
public class BinaryDecisionTreeClassifier implements Classifier{
	//private DataSet allData;
	private HashMap<Integer,String> featureMap;
	private Set<Integer> featureIndices;
	private DecisionTreeNode decisionTree;
	public int depthMax = Integer.MAX_VALUE;
	private boolean isRealData = true;	// defaults to true
	
	public void train(DataSet data) {
		if( data.getData().size() == 0 ){
			throw new RuntimeException("Tried to train without any data");
		}
		
		featureMap = data.getFeatureMap();
		featureIndices = data.getAllFeatureIndices();
		decisionTree = buildTree(data.getData(), new HashSet<Integer>(), depthMax);
	}
	
	/**
	 * Set the maximum height of the tree to be learned
	 * 
	 * @param depthMax the max depth of the tree
	 */
	public void setDepthLimit(int depthMax){
		this.depthMax = depthMax;
	}
	
	/**
	 * Set whether we are working with real values or categorical variables
	 * (This controls how we split)
	 * 
	 * @param depthMax the max depth of the tree
	 */
	public void setRealData(boolean isRealData){
		this.isRealData = isRealData;
	}
	
	/**
	 * Helper method for building the decision tree.
	 * 
	 * @param currentData the data (non-empty) to build the tree over
	 * @param usedFeatures the features that have been used already
	 * @param depthLimit the maximum depth we can build this tree
	 * @return the learned decision tree
	 */
	private DecisionTreeNode buildTree(ArrayList<Example> currentData, HashSet<Integer> usedFeatures, int depthLimit){
		DataMajority majority = getMajorityLabel(currentData);
				
		// base cases:
		// 1. they're all the same label
		// 2. we're out of features to examine
		if( majority.majorityCount == currentData.size() ||
			usedFeatures.size() == featureIndices.size() ||
			depthLimit == 0){
			return new DecisionTreeNode(majority.majorityLabel, majority.confidence);
		}else{
			// check if all examples have the same features
					
			// find the best feature that hasn't been used yet to split on
			double[] featureDetails = getBestFeatureIndex(currentData, usedFeatures);
			int bestFeature = (int) featureDetails[0];
			double threshold = featureDetails[1];
			
			// bestFeature != -1
			// split on the best feature
			ArrayList<Example>[] splits = splitData(currentData, bestFeature, threshold);
			
			// create a new decision tree node
			DecisionTreeNode node = new DecisionTreeNode(bestFeature, threshold);
			
			HashSet<Integer> featureCopy = (HashSet<Integer>)usedFeatures.clone();
			featureCopy.add(bestFeature);
			
			// left branch
			if( splits[0].size() == 0 ){
				node.setLeft(new DecisionTreeNode(majority.majorityLabel, majority.confidence));
			}else{
				node.setLeft(buildTree(splits[0],featureCopy, depthLimit-1));
			}
			
			// right branch
			if( splits[1].size() == 0 ){
				node.setRight(new DecisionTreeNode(majority.majorityLabel, majority.confidence));
			}else{
				node.setRight(buildTree(splits[1], featureCopy, depthLimit-1));
			}
			
			return node;
		}
	}
	
	/**
	 * Get the best feature to split on based on training error.
	 * 
	 * @param currentData the current set of examples
	 * @param usedFeatures which features have been used already and are NOT eligible for splitting on
	 * @return the index of the best feature
	 */
	private double[] getBestFeatureIndex(ArrayList<Example> currentData, HashSet<Integer> usedFeatures){
		int bestFeature = -1;
		double bestFeatureScore = 1.0; // lower is better for now
		double bestThreshold = -1.0;
		
		for( int featureIndex: featureIndices){
			if( !usedFeatures.contains(featureIndex) ){
				
				double[] errorInfo = averageTrainingError(currentData, featureIndex);
				double error = errorInfo[0];
				double threshold = errorInfo[1];
									
				if( error < bestFeatureScore ||
					(error == bestFeatureScore && featureIndex < bestFeature )){
					bestFeatureScore = error;
					bestFeature = featureIndex;
					bestThreshold = threshold;
				}
			}
		}
		
		return new double[] {bestFeature, bestThreshold};
	}
	
	/**
	 * Get the average training error on this data set if we split on featureIndex
	 * 
	 * @param data the current data
	 * @param featureIndex the feature we're considering splitting on
	 * @return the error
	 */
	private double[] averageTrainingError(ArrayList<Example> data, int featureIndex){		
		// sort the data
		data.sort((o1, o2) -> Double.compare(o1.getFeature(featureIndex),o2.getFeature(featureIndex)));
		double bestAccuracy = -1.0;
		double bestThreshold = 0.0;
		
		for (int i = 0; i < data.size(); i++) {
			double threshold = data.get(i).getFeature(featureIndex);
			ArrayList<Example>[] splits = splitData(data, featureIndex, threshold);
			
			int leftCount = splits[0].size() > 0 ? getMajorityLabel(splits[0]).majorityCount : 0;
			int rightCount = splits[1].size() > 0 ? getMajorityLabel(splits[1]).majorityCount : 0;
			
			double accuracy = (leftCount+rightCount)/(double)data.size();
			if (accuracy > bestAccuracy) {
				bestAccuracy = accuracy;
				bestThreshold = threshold;
			}
		}

		return new double[] {1-bestAccuracy, bestThreshold};
	}
	
	/**
	 * Split the data based on featureIndex
	 * 
	 * @param data the data to be split
	 * @param featureIndex the feature to split on
	 * @param threshold to split on
	 * @return the split of the data.  Entry 0 is the left branch data and entry 1 the right branch data.
	 */
	private ArrayList<Example>[] splitData(ArrayList<Example> data, int featureIndex, double threshold){
		// split the data based on this feature
		ArrayList<Example>[] splits = new ArrayList[2];
		splits[0] = new ArrayList<Example>();
		splits[1] = new ArrayList<Example>();
				
		for( Example d: data){
			double value = d.getFeature(featureIndex);
			
			if( value <= threshold ){
				splits[0].add(d);
			}else{
				splits[1].add(d);
			}
		}
		
		return splits;
	}
	
	public String toString(){
		return decisionTree.treeString(featureMap);
	}
	
	/**
	 * given the data, calculate the majority label and how many times it occurs in the data
	 * 
	 * @param data
	 * @return majority information from the data
	 */
	private DataMajority getMajorityLabel(ArrayList<Example> data){
		HashMapCounter<Double> counter = new HashMapCounter<Double>();
		
		for( Example d: data ){
			counter.increment(d.getLabel());
		}
		
		double maxLabel = 0.0;
		int maxCount = -1;
		
		for( Double key: counter.keySet() ){
			if( counter.get(key) > maxCount ){
				maxCount = counter.get(key);
				maxLabel = key;
			}
		}
		
		return new DataMajority(maxLabel, maxCount, ((double)maxCount)/data.size());
	}
		
	@Override
	public double classify(Example example) {
		return findLeaf(example).prediction();
	}
	
	@Override
	public double confidence(Example example) {
		return findLeaf(example).confidence();
	}
	
	/**
	 * Figure out which leaf this example falls into
	 * 
	 * @param example
	 * @return the leaf node
	 */
	private DecisionTreeNode findLeaf(Example example){
		DecisionTreeNode current = decisionTree;
		
		while( !current.isLeaf() ){
			int feature = current.getFeatureIndex();
			
			if( example.getFeature(feature) <= current.getThreshold() ){
				// go left
				current = current.getLeft();
			}else{
				current = current.getRight();
			}
		}
		
		return current;
	}
		
	/**
	 * A container class to allow us to return multiple values when calculting
	 * the majority label from a collection of data.
	 * 
	 * @author dkauchak
	 *
	 */
	private class DataMajority{
		public double majorityLabel;
		public int majorityCount;
		public double confidence;
		
		public DataMajority(double majorityLabel, int majorityCount, double confidence){
			this.majorityLabel = majorityLabel;
			this.majorityCount = majorityCount;
			this.confidence = confidence;
		}
	}	
}
