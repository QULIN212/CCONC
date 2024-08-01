package ceka.NWVNC;

import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.MultiNoisyLabelSet;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Utils;

public class NWVNC {
	private int K; 
	
	public Dataset nwvnc(Dataset dataset) throws Exception {
		
		K=9; 
		// Initialize the clean set and noise set
		Dataset cleanSet = dataset.generateEmpty();
		Dataset noiseSet = dataset.generateEmpty();
		for(int i = 0;i < dataset.getCategorySize();i++) {
			Category cate = dataset.getCategory(i);
			cleanSet.addCategory(cate);
			noiseSet.addCategory(cate);
		}
		
		// Maximum and minimum values for each feature
		double attValueMax[] = new double[dataset.numAttributes()];
		double attValueMin[] = new double[dataset.numAttributes()];
	
		// Find the maximum and minimum values for each feature
		for(int i = 0; i < dataset.numAttributes(); i++){
			if(i != dataset.classIndex()){
				if(dataset.attribute(i).isNumeric()){
					double[] attValue = new double[dataset.getExampleSize()];
					for (int j = 0; j < dataset.getExampleSize(); j++) {
						attValue[j] = dataset.getExampleByIndex(j).value(i);
					}	
					attValueMin[i] = attValue[Utils.minIndex(attValue)];
					attValueMax[i] = attValue[Utils.maxIndex(attValue)];
				}
			}
		}

		double[] w = new double[dataset.getExampleSize()];
		int[][] neighbor = new int[dataset.getExampleSize()][K]; 
	
		for(int n = 0; n < dataset.getExampleSize(); n++){
			Example e = dataset.getExampleByIndex(n);
			double[] distance = new double[dataset.getExampleSize()];
			int[] indexDistanceSort = new int[dataset.getExampleSize()];
			
			for(int i = 0; i < dataset.getExampleSize();i++) {
				Example e1 = dataset.getExampleByIndex(i);
				for(int j = 0; j < dataset.numAttributes();j++) {
					if(j != dataset.classIndex()) {
						if(e.isMissing(j) || e1.isMissing(j)) {
							distance[i] += 1;
						}else if (dataset.attribute(j).isNominal()){ 
							if (e.value(j) != e1.value(j)) {
								distance[i] += 1.0;
							}
						}else if (dataset.attribute(j).isNumeric()){ 
							double max_min = (attValueMax[j] - attValueMin[j]);
							if(max_min != 0.0){
								distance[i] += Math.pow((Math.abs(e.value(j)-e1.value(j)) / max_min),2);
							}
						}
					}
				}
				distance[i] = Math.sqrt(distance[i]);
			}
			indexDistanceSort = Utils.sort(distance);

			for(int k = 0;k < K;k++) {		
				neighbor[n][k] = indexDistanceSort[k];
			}
		
			int count = 0;
			MultiNoisyLabelSet mnls = e.getMultipleNoisyLabelSet(0);
			for(int j = 0;j < mnls.getLabelSetSize();j++) {
				if(e.getIntegratedLabel().getValue() == mnls.getLabel(j).getValue()) {
					count++;
				}
			}
			w[n] = (double)count / mnls.getLabelSetSize();;
		}
		
		for(int i = 0; i < dataset.getExampleSize();i++) {
			double countWeight = 0;
			double sameProb = 0;
			Example example = dataset.getExampleByIndex(i);
			
			for(int j = 0; j < K;j++) {
				Example e1 = dataset.getExampleByIndex(neighbor[i][j]);
				if(example.getIntegratedLabel().getValue() == e1.getIntegratedLabel().getValue())
					sameProb += w[neighbor[i][j]] ;
				countWeight += w[neighbor[i][j]];
			}
			sameProb /= countWeight ;
			if(sameProb < 0.5) {
				noiseSet.addExample(example);
			}else cleanSet.addExample(example);
		}
		
		Dataset correctDataset = correctNoise(cleanSet,noiseSet,dataset,neighbor,w);
		return correctDataset;
	}
	
	// correct
	public Dataset correctNoise(Dataset cleanSet, Dataset noiseSet, Dataset dataset, int[][] neighbor, double[] w) throws Exception {
		// Building three classifiers	
		Classifier classifier1 = new J48();
		classifier1.buildClassifier(cleanSet);
		Classifier classifier2 = new IBk();
		classifier2.buildClassifier(cleanSet);
		Classifier classifier3 = new NaiveBayes();
		classifier3.buildClassifier(cleanSet);
		// Correct instances of noise set
		for(int i = 0;i < noiseSet.getExampleSize();i++) {
			Example e = noiseSet.getExampleByIndex(i);
			int label1 = (int)classifier1.classifyInstance(e);
			int label2 = (int)classifier2.classifyInstance(e);
			int label3 = (int)classifier3.classifyInstance(e);
			if(label1 == label2 && label1 == label3 && label2 == label3 && label1 != e.getIntegratedLabel().getValue()) {
				e.getIntegratedLabel().setValue(label1);
				e.setTrainingLabel(label1);
			}
			cleanSet.addExample(e);
			
		}	
		return cleanSet;
	}
}
