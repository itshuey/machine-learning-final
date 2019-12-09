package ml.classifiers;

import java.util.Map;

/**
 * Class representing a node in a decision tree (both internal and leaf)
 * 
 * @author dkauchak
 *
 */
public class DecisionTreeNode {
	
	// whether or not this is a leaf
	private boolean leaf;

	// only applicable for leaves
	private double prediction = 0.0;
	private double confidence = 0.0;
	
	// only applicable if it's an internal node
	private int featureIndex;	// the index of the feature we're checking
	private double threshold;	// the threshold value we split on
	
	private DecisionTreeNode left;
	private DecisionTreeNode right;
	
	/**
	 * Create a leaf node with label "prediction"
	 * 
	 * @param prediction the prediction for this leaf node
	 */
	public DecisionTreeNode(double prediction, double confidence){
		leaf = true;
		this.prediction = prediction;
		this.confidence = confidence;
	}
	
	/**
	 * Create an internal node that splits on featureIndex
	 * 
	 * @param featureIndex
	 */
	public DecisionTreeNode(int featureIndex){
		leaf = false;
		this.featureIndex = featureIndex;
	}
	
	/**
	 * Create an internal node that splits on featureIndex with input threshold
	 * 
	 * @param featureIndex
	 */
	public DecisionTreeNode(int featureIndex, double threshold){
		leaf = false;
		this.featureIndex = featureIndex;
		this.threshold = threshold;
	}

	/**
	 * @return whether or not this node is a leaf
	 */
	public boolean isLeaf(){
		return leaf;
	}
	
	/**
	 * Only valid if this node is a leaf.
	 * 
	 * @return the prediction at this node
	 */
	public double prediction(){
		if( !leaf ){
			throw new RuntimeException("Can only call prediction on a leaf node");
		}
		
		return prediction;
	}
	
	/**
	 * Only valid if this node is a leaf.
	 * 
	 * @return the confidence at this node
	 */
	public double confidence(){
		if( !leaf ){
			throw new RuntimeException("Can only call prediction on a leaf node");
		}
		
		return confidence;
	}
	
	/**
	 * Set the left node of this node.
	 * 
	 * @param node
	 */
	public void setLeft(DecisionTreeNode node){
		this.left = node;
	}
	
	/**
	 * Get the left node
	 * 
	 * @return
	 */
	public DecisionTreeNode getLeft(){
		return left;
	}
	
	/**
	 * Set the right node of this node
	 * 
	 * @param node
	 */
	public void setRight(DecisionTreeNode node){
		this.right = node;
	}
	
	/**
	 * Get the right node
	 * 
	 * @return
	 */
	public DecisionTreeNode getRight(){
		return right;
	}
	
	/**
	 * Get the threshold
	 * 
	 * @return
	 */
	public double getThreshold(){
		return threshold;
	}
	
	
	/**
	 * Get the feature index that this internal node splits on
	 * 
	 * @return
	 */
	public int getFeatureIndex(){
		if( leaf ){
			throw new RuntimeException("getFeatureIndex can only be called on internal nodes");
		}
		
		return featureIndex;
	}
	
	/**
	 * Get a formatted string representation of this DecisionTreeNode and all nodes below it.
	 * This basic version only prints out feature indices.
	 * 
	 * @return
	 */
	public String treeString(){
		return treeStringHelper("  ", null);
	}
	
	/**
	 * Get a formatted string representation of this DecisionTreeNode and all nodes below it.
	 * This version will put in the actual feature names based on the mapping supplied.
	 * 
	 * @param featureMap
	 * @return
	 */
	public String treeString(Map<Integer,String> featureMap){
		return treeStringHelper("  ", featureMap);
	}

	/**
	 * @param spaces the amount of space to indent the child nodes
	 * @param headers
	 * @return
	 */
	private String treeStringHelper(String spaces, Map<Integer,String> headers){
		if( leaf ){
			return "predict: " + Double.toString(prediction);
		}else{
			String featureString;
			
			if( headers == null ){
				featureString = Integer.toString(featureIndex);
			}else{
				featureString = headers.get(featureIndex) + " <= " + threshold;
			}
			
			return "(" + featureString + "\n" + 
		            spaces + left.treeStringHelper(spaces + "  ", headers) + "\n" +
					spaces + right.treeStringHelper(spaces + "  ", headers) + ")";
		}
	}
}
