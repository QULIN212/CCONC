package ceka.CWVNC;

import java.util.ArrayList;
import java.util.HashMap;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Utils;


public class CWVNC {
	
	private static final String NAME = "CWVNC";

	public Dataset CWVNC(Dataset dataset) throws Exception {

		HashMap<String,Dataset>subDataset = new HashMap<String, Dataset>();
		Dataset tempTrain = copyDataset(dataset);
		for(int eId = 0; eId < tempTrain.getExampleSize(); eId++) {
			Example example = tempTrain.getExampleByIndex(eId);
			ArrayList<String> workerList = example.getWorkerIdList();
			for(int wId = 0; wId < workerList.size(); wId++) {
				String w1 = workerList.get(wId);
				if(!subDataset.containsKey(w1)) {
					Dataset temp_dataset = new Dataset(tempTrain, 0);
					subDataset.put(w1, temp_dataset);
				}
				subDataset.get(w1).addExample(example);
			}
		}
	
		HashMap<String, NaiveBayes> worker2NB = new HashMap<String, NaiveBayes>();
		for (String key : subDataset.keySet()) {
			if (subDataset.get(key).getExampleSize() > 0.1 * dataset.getExampleSize()) {
				Dataset tempData = subDataset.get(key);
				for (int j = 0; j < tempData.getExampleSize(); j++) {
					Example example = tempData.getExampleByIndex(j);
					int value = example.getNoisyLabelByWorkerId(key).getValue();
					example.setTrainingLabel(value);
				}
				NaiveBayes tempNB = new NaiveBayes();
				tempNB.buildClassifier(tempData);
				worker2NB.put(key, tempNB);
			}
		}

		HashMap<String,HashMap<Integer, Double>>weight = new HashMap<>();
		
		HashMap<Integer, Double> class2Num = new HashMap<Integer, Double>();
		HashMap<Integer, Double> class2Worker = new HashMap<Integer, Double>();
		for(int i=0; i<dataset.getCategorySize(); i++) {
			class2Num.put(i, 0.0);
			class2Worker.put(i, 0.0);
		}
		
		
		for(String key : worker2NB.keySet()) {
			if(!weight.containsKey(key)) {
				HashMap<Integer, Double> tempWeightHashMap = new HashMap<Integer, Double>();
				for(int cid=0; cid<dataset.getCategorySize(); cid++) {
					tempWeightHashMap.put(cid, 0.0);
				}
				weight.put(key, tempWeightHashMap);
			}
		}
	
		
		for(int j=0; j<dataset.getExampleSize(); j++) {
			Example tempExample = dataset.getExampleByIndex(j);
			int tempClass = tempExample.getIntegratedLabel().getValue();  
			class2Num.replace(tempClass, class2Num.get(tempClass) + 1.0); 
			
			for (String key : weight.keySet()) {
				if(tempExample.getNoisyLabelByWorkerId(key) != null) {
					Label tempLabel = tempExample.getNoisyLabelByWorkerId(key);
					class2Worker.replace(tempLabel.getValue(), class2Worker.get(tempLabel.getValue()) + 1); 
					HashMap<Integer, Double> tempW = weight.get(key);
					if(tempLabel.getValue() == tempExample.getIntegratedLabel().getValue()) {  
						weight.get(key).replace(tempLabel.getValue(), tempW.get(tempLabel.getValue()) + 1.0); 
					}
				}
			}
		}
		
		for(String key : weight.keySet()) {
			HashMap<Integer, Double> tempW = weight.get(key);
			for(Integer key2 : tempW.keySet()) {
				tempW.replace(key2, tempW.get(key2)/class2Num.get(key2));
				
			}
			weight.replace(key, tempW);
		}
		
		
		for(int i = 0;i < dataset.getExampleSize();i++) {
			Example example = dataset.getExampleByIndex(i);
			double[] classCounts = new double[dataset.getCategorySize()];
			for (String key : worker2NB.keySet()) {		
				if(example.getNoisyLabelByWorkerId(key) != null) {
					double[] tempProbs = worker2NB.get(key).distributionForInstance(example);
					int tempLabel = example.getNoisyLabelByWorkerId(key).getValue();
					classCounts[tempLabel] += tempProbs[tempLabel] * weight.get(key).get(tempLabel);

					
				}
			}
			
			int index = Utils.maxIndex(classCounts);
			if(index != example.getIntegratedLabel().getValue()) {
				Label label = new Label(null, String.valueOf(index), example.getId(), NAME);
				example.setIntegratedLabel(label);
			}
		}

		dataset.assignIntegeratedLabel2WekaInstanceClassValue();
		return dataset;
	}
		
	public static Dataset copyDataset(Dataset dataset) {
		Dataset copyDataset = new Dataset(dataset, 0);
		for (int k = 0; k < dataset.getExampleSize(); k++) {
			Example example = dataset.getExampleByIndex(k);
			copyDataset.addExample(example);
		}
		for (int k = 0; k < dataset.getCategorySize(); k++) {
			Category category = dataset.getCategory(k);
			copyDataset.addCategory(category);
		}
		for (int k = 0; k < dataset.getWorkerSize(); k++) {
			Worker worker = dataset.getWorkerByIndex(k);
			copyDataset.addWorker(worker);
		}
		return copyDataset;
	}
	
}
