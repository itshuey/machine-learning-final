package ml.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * A collections of examples representing an entire data set.
 * 
 * @author dkauchak
 */
public class DataSet {
	private ArrayList<Example> data = new ArrayList<Example>(); // the data/examples in this data set
	// the mapping from feature indices to the name of the feature
	private HashMap<Integer, String> featureMap = new HashMap<Integer, String>();
	private HashSet<Double> labels = new HashSet<Double>();
	
	// some constants for different file types
	public static final int CSVFILE = 0;
	public static final int TEXTFILE = 1;
	
	private int featureIndex;
	private int biasFeature = -1;

	/**
	 * Parsing the abalone data set
	 * 
	 * @param filename the location of the file
	 */
	public DataSet(String filename, String dataname){

		try {
			BufferedReader in = new BufferedReader(new FileReader(filename));

			if (dataname.equals("abalone")) {
				String line;
				while ((line = in.readLine()) != null) {
					Example current = new Example();
					String[] data = line.split(",");
					
					double label = Double.parseDouble(data[data.length-1]) < 9 ? -1.0 : 1.0;
					current.setLabel(label);
					
					for (int i=data.length-2; i > 0; i--) 
						current.setFeature(i, Double.parseDouble(data[i]));
	
					double sex;
					if (data[0].equals("M")) 		sex = 1.0;
					else if (data[0].equals("F")) 	sex = -1.0;
					else 							sex = 0.0;
					
					current.setFeature(0, sex);
					addData(current);
				}
				
				HashMap<Integer, String> features = new HashMap<>();
				features.put(0, "Sex");
				features.put(1, "Length");
				features.put(2, "Diameter");
				features.put(3, "Height");
				features.put(4, "Whole weight");
				features.put(5, "Shucked weight");
				features.put(6, "Viscera weight");
				features.put(7, "Shell weight");
				
				featureMap = features;
			} 
			
			else if (dataname.equals("ionosphere")){
				String line;
				while ((line = in.readLine()) != null) {
					Example current = new Example();
					String[] data = line.split(",");
					
					double label = data[data.length-1].equals("g")  ? 1.0 : -1.0;
					current.setLabel(label);
					
					for (int i=data.length-2; i >= 0; i--) 
						current.setFeature(i, Double.parseDouble(data[i]));

					addData(current);
				}
				
				HashMap<Integer, String> features = new HashMap<>();
				for (int i = 0; i <= 33; i++) {
					features.put(i, "Feature " + i);
				}	
				featureMap = features;
			} 
			
			else if (dataname.equals("cleveland")){
				String line;
				while ((line = in.readLine()) != null) {
					Example current = new Example();
					String[] data = line.split(",");
					
					double label = data[data.length-1].equals("0")  ? 1.0 : -1.0;
					current.setLabel(label);
					
					for (int i=data.length-2; i >= 0; i--) {
						double entry = data[i].equals("?") ? 0.0 : Double.parseDouble(data[i]);
						current.setFeature(i, entry);
					}

					addData(current);
				}
				
				HashMap<Integer, String> features = new HashMap<>();
				features.put(0, "age");
				features.put(1, "sex");
				features.put(2, "cp");
				features.put(3, "testbps");
				features.put(4, "chol");
				features.put(5, "fbs");
				features.put(6, "restecg");
				features.put(7, "thalach");
				features.put(8, "exang");
				features.put(9, "oldpeak");
				features.put(10, "slope");
				features.put(11, "ca");
				features.put(12, "thal");
	
				featureMap = features;
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * Create a new data set.  
	 * 
	 * @param filename the location of the file
	 * @param fileType what type of file, using the class defined constants (e.g. CSVFILE)
	 */
	public DataSet(String filename, int fileType){
		if( fileType == CSVFILE ){
			// figure out how many columns there are then call
			try {
				BufferedReader in = new BufferedReader(new FileReader(filename));

				// ignore any lines at the beginning that start with #
				String line = in.readLine();

				while( line.startsWith("#")){
					line = in.readLine();
				}
				
				// parse the headers
				String[] headers = line.split(",");
				int labelIndex = headers.length-1;					
				featureIndex = 0;
					
				for( int i = 0; i < headers.length; i++ ){
					if( i != labelIndex ){
						featureMap.put(featureIndex, headers[i]);
						featureIndex++;
					}
				}
					
				CSVDataReader reader = new CSVDataReader(in, labelIndex);
				initialize(reader);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}else if( fileType == TEXTFILE ){
			TextDataReader reader = new TextDataReader(filename);
			initialize(reader);
			featureMap = reader.getFeatureMap();
		}
	}
	
	/**
	 * Read all of the data from the reader and populate this dataset.
	 * 
	 * @param reader
	 */
	private void initialize(Iterator<Example> reader){			
		while( reader.hasNext()){
			Example next = reader.next();				
			data.add(next);
			labels.add(next.getLabel());
		}
	}
		
	/**
	 * Constructs a new empty dataset (i.e. no examples) with the features
	 * specified in the featuremap
	 * 
	 * @param s
	 */
	public DataSet(HashMap<Integer, String> featureMap){
		this.featureMap = new HashMap<Integer, String>(featureMap);
		
		// figure out what the largest feature is
		int maxIndex = -1;
		
		for( Integer index: featureMap.keySet() ){
			if( index > maxIndex ){
				maxIndex = index;
			}
		}
		
		featureIndex = maxIndex+1;
	}
	
	/**
	 * Get the mapping from feature indices to feature names.  This is
	 * mostly useful when trying to print out the final models.
	 * 
	 * @return feature map
	 */
	public HashMap<Integer,String> getFeatureMap(){
		return featureMap;
	}	
	
	/**
	 * Get the examples associated with this data set
	 * 
	 * @return the examples
	 */
	public ArrayList<Example> getData(){
		return data;
	}
	
	/**
	 * Add all of the examples in addMe to this data set.
	 * Note: this does NOT change the feature map for this
	 * data set so this should only be used to add examples that
	 * have the same features that the data set was already initialized
	 * with.
	 * 
	 * @param addMe
	 */
	public void addData(ArrayList<Example> addMe){
		for( Example e: addMe ){
			data.add(e);
			labels.add(e.getLabel());
		}
	}

	/**
	 * Add example e to this data set.
	 * Note: this does NOT change the feature map for this
	 * data set so this should only be used to add examples that
	 * have the same features that the data set was already initialized
	 * with.
	 * 
	 * @param addMe
	 */
	public void addData(Example e){
		data.add(e);
		labels.add(e.getLabel());
	}
	
	/**
	 * Get all of the feature indices that are contained in this
	 * data set.
	 * 
	 * @return
	 */
	public Set<Integer> getAllFeatureIndices(){
		return featureMap.keySet();
	}
	
	/**
	 * Get all the labels in this data set
	 * 
	 * @return the labels
	 */
	public Set<Double> getLabels(){
		return labels;
	}
	
	/**
	 * Split this data set into two data sets of size:
	 * - total_size * fraction
	 * - total_size - (total_size*fraction)
	 * 
	 * @param fraction the proportion to allocated to the first data set in the split
	 * @return a split of the data
	 */
	public DataSetSplit split(double fraction){
		ArrayList<Example> newdata = (ArrayList<Example>)data.clone();
		Collections.shuffle(newdata, new Random(System.nanoTime()));
		
		ArrayList<Example> train = new ArrayList<Example>();
		ArrayList<Example> test = new ArrayList<Example>();
		
		int trainSize = (int)Math.floor(data.size()*fraction);
		
		for( int i = 0; i < newdata.size(); i++ ){
			if( i < trainSize ){
				train.add(newdata.get(i));
			}else{
				test.add(newdata.get(i));
			}
		}
		
		DataSet dTrain = new DataSet(featureMap);
		dTrain.addData(train);
		
		DataSet dTest = new DataSet(featureMap);
		dTest.addData(test);

		return new DataSetSplit(dTrain, dTest);
	}	
	
	/**
	 * Get a cross-validation of this data set with num splits.  The
	 * data is split WITHOUT changing the order or the data.
	 * 
	 * @param num
	 * @return
	 */
	public CrossValidationSet getCrossValidationSet(int num){
		return new CrossValidationSet(this, num);
	}
	
	/**
	 * Get a cross-validation of this data set with num splits.  The
	 * data is randomized before splitting (though the data in this
	 * data set itself will not change).
	 * 
	 * @param num
	 * @return
	 */
	public CrossValidationSet getRandomCrossValidationSet(int num){
		return new CrossValidationSet(this, num, true);
	}
	
	/**
	 * Create a complete copy of this dataset with a bias added
	 * to the dataset, including all of the examples in the dataset. "this" DataSet
	 * remains unchanged.  To add a bias to additional examples outside of 
	 * the dataset (e.g. for classifying), call addBiasFeature on the *new* dataset.
	 * 
	 * @param d
	 * @return
	 */
	public DataSet getCopyWithBias(){
		// get an empty copy
		DataSet biased = new DataSet(getFeatureMap());
		
		// set the bias index for the new dataset
		biased.biasFeature = featureIndex;
		
		// add the bias to the feature mapping
		biased.featureMap.put(biased.biasFeature, "bias");
		
		// iterate through all of the features and add the bias using addBiasFeature from
		// the *new* dataset
		for( Example e: getData() ){
			biased.addData(biased.addBiasFeature(e));
		}
		
		return biased;
	}
	
	/**
	 * Takes as input an example that is *not* already part of this DataSet
	 * and creates a copy of that Example with a bias added
	 * 
	 * @param e
	 * @return new Example with the bias added
	 */
	public Example addBiasFeature(Example e){
		if( biasFeature == -1 ){
			throw new RuntimeException("Called DataSet.addBiasFeature on a DataSet that didn't includ a bias");
		}
		
		if( e.getFeatureSet().contains(biasFeature) ){
			System.err.println("Warning: adding bias feature to an example that already has it.  This *could* be an error");
		}
		
		Example biased = new Example(e);
		biased.addFeature(biasFeature, 1.0);
		
		return biased;
	}
}
