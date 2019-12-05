package ml.classifiers;

import java.util.Map;
import java.util.Random;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import ml.data.DataSet;
import ml.data.Example;

/**
 * TwoLayerNN is a two-layer neural net classifier
 * 
 * @author huey
 */
public class TwoLayerNN implements Classifier {
	private int numHiddenNodes;
	private int numIterations;
	private double eta;
	
	public static int TANH = 1;
	
	private Map<Integer, Map<Integer, Double>> featureWeights;
	private Map<Integer, Double> hiddenLayerWeights;
	
	private DataSet train;
	private DataSet test;
	
	//////////////////////
	// SETUP METHODS
	/////////////////////
	
	/**
	 * TwoLayerNN constructs a two layer neural network
	 * 
	 * @param hiddenNodes, number of hidden nodes
	 */
	public TwoLayerNN(int hiddenNodes) {
		numHiddenNodes = hiddenNodes;
		numIterations = 200;
		eta = 0.1;
	}
	
	/**
	 * setEta sets eta, the learning rate, for the net
	 * 
	 * @param eta
	 */
	public void setEta(double eta) {
		this.eta = eta;
	}
	
	/**
	 * setIterations sets the number of iterations for training
	 * 
	 * @param iuterations
	 */
	public void setIterations(int iterations) {
		this.numIterations = iterations;
	}

	//////////////////////
	// CORE METHODS
	/////////////////////
	
	/**
	 * Trains the neural net on input data
	 * 
	 * @param data
	 */
	@Override
	public void train(DataSet data) {
		// get data with bias
		DataSet dataWithBias = data.getCopyWithBias();
		initializeWeights(dataWithBias);
		
		for (int iter = 0; iter < numIterations; iter++) {

			// iterate through examples
			for (Example e : dataWithBias.getData()) {
				
				double prediction = predict(e);
				double label = e.getLabel();
				double error = -(label - prediction);
				double slope = applyActivationFunctionDerivative(TANH, predict(e, true));
				Map<Integer, Double> hiddenLayerValues = getHiddenLayerValues(e, true);
				Map<Integer, Double> originalHiddenLayerWeights = new HashMap<>(hiddenLayerWeights);
				
				// update each hidden layer weight
				for (int node : hiddenLayerWeights.keySet()) {
					double hiddenLayerValue = node == -1 ? 1.0 : applyActivationFunction(TANH, hiddenLayerValues.get(node));
					double hiddenLayerUpdate = error * slope * hiddenLayerValue;
					hiddenLayerWeights.put(node, hiddenLayerWeights.get(node) - eta * hiddenLayerUpdate);
					
					// don't backpropogate the bias node
					if (node == -1) continue;
					for (int feature : featureWeights.keySet()) {

						Map<Integer, Double> featureWeight = featureWeights.get(feature);
						double input = e.getFeature(feature);
						double originalWeight = originalHiddenLayerWeights.get(node);
						double slopewx = applyActivationFunctionDerivative(TANH, hiddenLayerValues.get(node));
	
						double update = error * slope * originalWeight * slopewx * input;
						featureWeight.put(node, featureWeight.get(node) - eta * update);
					}
				}
			}
		}
	}
	
	
	/**
	 * Classifies an input example with the neural net
	 * Assumes you have already trained your classifier
	 * 
	 * @param example
	 */
	@Override
	public double classify(Example example) {
		return predict(example) > 0 ? 1 : -1;
	}

	/**
	 * Gets the confidence of an input example with the neural net
	 * Assumes you have already trained your classifier
	 * 
	 * @param example
	 */
	@Override
	public double confidence(Example example) {
		return Math.abs(predict(example));
	}
	
	//////////////////////
	// HELPER METHODS
	/////////////////////
	
	/**
	 * Returns the value of an input example with the current weights
	 * 
	 * @param example
	 */
	public double predict(Example example) {
		return predict(example, false);
	}
	
	/**
	 * Helper function that returns value of an input example with the current weights
	 * 
	 * @param returnRawValue: if true, predict does not apply the activation function to the output
	 * @param example
	 */
	private double predict(Example example, boolean returnRawValue) {
		
		// Create map to hold hidden layer values and bias
		Map<Integer, Double> hiddenLayerValues = getHiddenLayerValues(example, false);
		
		// Calculate raw output
		double output = getOutputValue(hiddenLayerValues, example);
		
		// Either return raw value or value with activation function
		return returnRawValue ? output : applyActivationFunction(output);
	}
	
	/**
	 * Calculates the hidden layer values for an input example
	 * 
	 * @param example
	 * @return Map of (hiddenLayer -> value) pairs
	 */
	private Map<Integer, Double> getHiddenLayerValues(Example example, boolean getRawValues) {
		// Create map to hold hidden layer values and bias
		Map<Integer, Double> hiddenLayerValues = new HashMap<>();
		hiddenLayerValues.put(-1, 1.0);
	
		// For each feature, calculate contribution to each hidden node
		for (int feature : example.getFeatureSet()) {
			double featureValue = example.getFeature(feature);
			Map<Integer, Double> featureToHiddenLayers = featureWeights.get(feature);
		
			for (int node = 0; node < numHiddenNodes; node++) {
				double currentHiddenLayerValue = hiddenLayerValues.getOrDefault(node, 0.0);
				hiddenLayerValues.put(node, currentHiddenLayerValue + featureToHiddenLayers.get(node) * featureValue);
			}
		}
		
		if (getRawValues) return hiddenLayerValues;
		
		// Apply activation function
		for (int node : hiddenLayerValues.keySet()) {
			if (node == -1) continue; // don't process bias term
			hiddenLayerValues.put(node, applyActivationFunction(hiddenLayerValues.get(node)));
		}
		
		return hiddenLayerValues;
	}
	
	/**
	 * Initializes feature weight gradient to be all 0's
	 *
	 * @param input DataSet
	 */
	private Map<Integer, Map<Integer, Double>> initializeFeatureWeightGradient(DataSet data) {
		Map<Integer, Map<Integer, Double>> output = new HashMap<>();
		for (int feature : data.getAllFeatureIndices()) {
			Map<Integer, Double> temp = new HashMap<>();
			for (int i = 0; i < numHiddenNodes; i++) {
				temp.put(i, 0.0);
			}
			output.put(feature, temp);
		}
		return output;
	}
	
	/**
	 * Initializes hidden weight gradient to be all 0's
	 *
	 * @param input DataSet
	 */
	private Map<Integer, Double> initializeHiddenWeightGradient() {
		Map<Integer, Double> output = new HashMap<>();
		for (int i = -1; i < numHiddenNodes; i++) {
			output.put(i, 0.0);
		}
		return output;
	}
	
	/**
	 * Calculates the raw output value for an input example 
	 * 		with respect to the input map of hidden layer values
	 * 
	 * @param example, hiddenLayerValues 
	 * @return output
	 */
	private double getOutputValue(Map<Integer, Double> hiddenLayerValues, Example example) {
		double output = 0.0;
		for (int hiddenNode : hiddenLayerValues.keySet()) {
			output += hiddenLayerValues.get(hiddenNode) * hiddenLayerWeights.get(hiddenNode);
		}
		return output;
	}
	
	/**
	 * InitializeWeights initializes random weights for the input layer
	 * 		and hidden layer between -0.1 and 0.1
	 * 
	 * @param data
	 */
	private void initializeWeights(DataSet data) {
		// create initial maps
		featureWeights = new HashMap<>();
		hiddenLayerWeights = new HashMap<>();
		Random rand = new Random();

		// initialize first layer weights
		for (int feature : data.getAllFeatureIndices()) {
			Map<Integer, Double> initialWeights = new HashMap<Integer, Double>();
			for (int node = 0; node < numHiddenNodes; node++) {
				initialWeights.put(node, getRandomInitialWeight(rand));
			}
			featureWeights.put(feature, initialWeights);
		}
		
		// initialize hidden layer weights
		for (int node = -1; node < numHiddenNodes; node++) {
			// -1 weight will correspond to the bias
			hiddenLayerWeights.put(node, getRandomInitialWeight(rand));
		}
	}
	
	/**
	 * Helper function to return a random intial weight
	 * Value will be between -0.1 and 0.1
	 * 
	 * @return random double
	 */
	private static double getRandomInitialWeight(Random rand) {
		return (rand.nextDouble() - 0.5) / 5;
	}
	
	/**
	 * Helper function that applies default tanh activation function
	 * 
	 * @return output
	 */
	private static double applyActivationFunction(double value) {
		return applyActivationFunction(TANH, value);
	}
	
	/**
	 * Helper function that applies an activation function
	 * 
	 * @return output
	 */
	private static double applyActivationFunction(int activationFunction, double value) {
		if (activationFunction == TANH) {
			return Math.tanh(value);
		} else {
			return 0.0;
		}
	}
	
	/**
	 * Helper function that applies the derivative of the activation function
	 * 
	 * @return output
	 */
	private static double applyActivationFunctionDerivative(int activationFunction, double value) {
		if (activationFunction == TANH) {
			return 1 - Math.pow(Math.tanh(value),2);
		} else {
			return 0.0;
		}
	}
	
	/**
	 * Trains the neural net on input data,
	 * updating once each iteration
	 * 
	 * @param data
	 */
	public void altTrain(DataSet data) {
		// get data with bias
		DataSet dataWithBias = data.getCopyWithBias();
		initializeWeights(dataWithBias);
		test = test.getCopyWithBias();
		
		double[] sse = new double[200], tre = new double[200], tse = new double[200];
		
		for (int iter = 0; iter < numIterations; iter++) {
			
			// initialize storage for updates
			Map<Integer, Map<Integer, Double>> featureWeightUpdates = initializeFeatureWeightGradient(dataWithBias);
			Map<Integer, Double> hiddenLayerWeightUpdates = initializeHiddenWeightGradient();

			Collections.shuffle(dataWithBias.getData());
			// iterate through examples
			for (Example e : dataWithBias.getData()) {
				
				double prediction = predict(e);
				double label = e.getLabel();
				double error = -(label - prediction);
				double slope = applyActivationFunctionDerivative(TANH, predict(e, true));
				Map<Integer, Double> hiddenLayerValues = getHiddenLayerValues(e, true);
				
				// update each hidden layer weight
				for (int node : hiddenLayerWeights.keySet()) {
					double hiddenLayerValue = node == -1 ? 1.0 : applyActivationFunction(TANH, hiddenLayerValues.get(node));
					double hiddenLayerUpdate = error * slope * hiddenLayerValue;
					hiddenLayerWeightUpdates.put(node, hiddenLayerWeightUpdates.get(node) + hiddenLayerUpdate);
					
					// don't backpropogate the bias node
					if (node == -1) continue;
					for (int feature : featureWeights.keySet()) {

						Map<Integer, Double> featureWeight = featureWeightUpdates.get(feature);
						double input = e.getFeature(feature);
						double current = featureWeight.get(node);
						double originalWeight = hiddenLayerWeights.get(node);
						double slopewx = applyActivationFunctionDerivative(TANH, hiddenLayerValues.get(node));
	
						double update = error * slope * originalWeight * slopewx * input;
						featureWeight.put(node, current + update);
					}
				}
			}
			
			for (int feature : dataWithBias.getAllFeatureIndices()) {
				Map<Integer, Double> toUpdate = featureWeights.get(feature);
				Map<Integer, Double> updateFrom = featureWeightUpdates.get(feature);
				for (int node : toUpdate.keySet()) {
					toUpdate.put(node, toUpdate.get(node) - eta * updateFrom.get(node));
				}
			}
			
			for (int node : hiddenLayerWeights.keySet()) {
				double update = hiddenLayerWeightUpdates.get(node);
				hiddenLayerWeights.put(node, hiddenLayerWeights.get(node) - eta * update);
			}
			
			sse[iter] = sumOfSquareError(dataWithBias);
			tre[iter] = getAccuracy(dataWithBias);
			tse[iter] = getAccuracy(test);
		}
		
		for (double s : sse) {
			System.out.println(s);
		}
		
		System.out.println();
		for (double s : tre) {
			System.out.println(s);
		}
		
		System.out.println();
		for (double s : tse) {
			System.out.println(s);
		}
	}
	
	
	//////////////////////
	// UTILITY METHODS
	/////////////////////
	
	/**
	 * Calculates the sum of squares error for an input dataset 
	 * 		with respect to the current weights
	 * 
	 * @param data
	 * @return error
	 */
	public double sumOfSquareError(DataSet data) {
		double error = 0.0;
		for (Example e : data.getData()) {
			error += Math.pow(e.getLabel() - predict(e), 2);
		}
		return error;
	}
	
	/**
	 * Calculates the accuracy of a data set with the classifier
	 * 
	 * @param data
	 * @return accuracy
	 */
	public double getAccuracy(DataSet data) {
		int correct = 0, total = 0;	
		
		for (Example e : data.getData()) {
			if (classify(e) == e.getLabel()) correct++;
			total++;
		}
		
		return (double) correct / total;
	}
	
	/**
	 * Utility function that allows you to manually set the feature weights
	 * 
	 */
	public void setFeatureWeights(Map<Integer, Map<Integer, Double>> featureWeights) {
		this.featureWeights = featureWeights;
	}
	
	/**
	 * Utility function that allows you to manually set the hidden layer weights
	 * 
	 */
	public void setHiddenLayerWeights(Map<Integer, Double> hiddenLayerWeights) {
		this.hiddenLayerWeights = hiddenLayerWeights;
	}
	
	/**
	 * Utility function that allows you to set the training data
	 * 
	 */
	public void setTrainingData(DataSet trainingData) {
		this.train = trainingData;
	}
	
	/**
	 * Utility function that allows you to set the test data
	 * 
	 */
	public void setTestData(DataSet testData) {
		this.test = testData;
	}
	
	/**
	 * Utility function that prints out all the feature weights 
	 * 		and hidden layer weights
	 * 
	 */
	public void printWeights() {
		System.out.println("Feature Weights:");
		for (int feature : featureWeights.keySet()) {
			System.out.println("Feature " + feature);
			Map<Integer,Double> featureWeighting = featureWeights.get(feature);
			for(int hiddenNode : featureWeighting.keySet()) {
				System.out.println("To hidden node " + hiddenNode + ": " 
					+ featureWeighting.get(hiddenNode));
			}
			System.out.println();
		}
		
		System.out.println("Hidden Layer Weights:");
		for (int hiddenNode : hiddenLayerWeights.keySet()) {
			System.out.println("Node " + hiddenNode + ": " + hiddenLayerWeights.get(hiddenNode));
		}	
		System.out.println();
	}

}
