package ceka.WSNC;

import ceka.core.*;
import ceka.utils.DescendingElement;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;


public class WSNC {
	public WSNC() {
		
	}

	public Dataset WSNC(Dataset dataset, int m_KNN) throws Exception {

		int numExample = dataset.getExampleSize();
		int numWorker = dataset.getWorkerSize();
		int numClass = dataset.numClasses();
		HashMap<String, Integer> nclassMap=new HashMap<>(); 

		Dataset noiseSet = dataset.generateEmpty();
		Dataset cleanSet = dataset.generateEmpty();

		for (int i = 0; i < dataset.getCategorySize(); i++) {
			Category cate = dataset.getCategory(i);
			noiseSet.addCategory(cate.copy());
			cleanSet.addCategory(cate.copy());
		}

		//The similarity between each worker and other workers, sorted in descending order
		HashMap<String, ArrayList<DescendingElement<String>>> hashMap = new HashMap<>();

		//Calculate the similarity between other workers and each worker
		for (int i = 0; i < numWorker; i++) {
			Worker worker1 = dataset.getWorkerByIndex(i);
			double[] sum_sim = new double[numWorker]; 
			double[] workerSim = new double[numWorker]; 
			for(int r=0;r<numWorker;r++) {
				Worker worker2=dataset.getWorkerByIndex(r);
				for(int j=0; j<numExample ;j++) {
					Example example=dataset.getExampleByIndex(j);
					Label label1=example.getNoisyLabelByWorkerId(worker1.getId());
					Label label2=example.getNoisyLabelByWorkerId(worker2.getId());
					if(label1!=null && label2!=null) {
						sum_sim[r]++;
						if(label1.getValue()==label2.getValue()) {
							workerSim[r]++;
						}
					}
				}
			}
			
			// Calculate the similarity between this worker and other workers
			ArrayList<DescendingElement<String>> workSim = new ArrayList<DescendingElement<String>>();
			for (int j = 0; j < numWorker; j++) {
				DescendingElement<String> element = new DescendingElement<String>();
				if (i == j) {
					workerSim[j]=1; 
				} else {
					if (sum_sim[j] < numExample * 0.1) {
						workerSim[j] = 0;				
					} else {
						workerSim[j] = workerSim[j] / sum_sim[j];
					}
				}
				element.setData(dataset.getWorkerByIndex(j).getId()); // ����index
				element.setKey(workerSim[j]); //���ƶ�
				workSim.add(element);
			}
			Collections.sort(workSim);
			hashMap.put(worker1.getId(), workSim);
		}

		double attValueMax[] = new double[dataset.numAttributes()];
		double attValueMin[] = new double[dataset.numAttributes()];
		// Find the maximum and minimum values for each attribute
		for (int i = 0; i < dataset.numAttributes(); i++) {
			if (i != dataset.classIndex()) {
				if (dataset.attribute(i).isNumeric()) {
					double[] attValue = new double[numExample];
					for (int j = 0; j < numExample; j++) {
						attValue[j] = dataset.getExampleByIndex(j).value(i);
					}
					attValueMin[i] = attValue[Utils.minIndex(attValue)];
					attValueMax[i] = attValue[Utils.maxIndex(attValue)];
				}
			}
		}
		// Find K similar instances for each instance,HEOM
		for (int i = 0; i < numExample; i++) {
			int[] indexDistanceSort = new int[numExample];
			double[] distance = new double[numExample];
			Example e = dataset.getExampleByIndex(i);
			for (int j = 0; j < numExample; j++) {
				Example e1 = dataset.getExampleByIndex(j);
				for (int k = 0; k < dataset.numAttributes(); k++) {
					if (k != dataset.classIndex()) {
						if (e.isMissing(k) || e1.isMissing(k)) {
							distance[j] += 1;
						} else if (dataset.attribute(k).isNominal()) {
							if (e.value(k) != e1.value(k)) {
								distance[j] += 1.0;
							}
						} else if (dataset.attribute(k).isNumeric()) {
							double max_min = (attValueMax[k] - attValueMin[k]);
							if (max_min != 0.0) {
								distance[j] += Math.pow((Math.abs(e.value(k) - e1.value(k)) / max_min), 2);
							}
						}
					}
				}
				distance[j] = Math.sqrt(distance[j]);

			}
			//sort the distances
			indexDistanceSort = Utils.sort(distance);

			//caculate each label's quality
			MultiNoisyLabelSet eLabelSet=e.getMultipleNoisyLabelSet(0);
			double[] labelCon=new double[eLabelSet.getLabelSetSize()]; 
			double[] label_sum=new double[eLabelSet.getLabelSetSize()]; 
			for(int j=0;j<eLabelSet.getLabelSetSize();j++) {
				Label label1=eLabelSet.getLabel(j); 
				String workerid = label1.getWorkerId();
				for(int rk=0;rk<m_KNN; rk++) {
					String simuWorkerId = hashMap.get(workerid).get(rk).getData(); 
					double simu = hashMap.get(workerid).get(rk).getKey();
					for(int ik=0; ik<m_KNN; ik++) {
						Example example=dataset.getExampleByIndex(indexDistanceSort[ik]);
						Label label2=example.getNoisyLabelByWorkerId(simuWorkerId);
						if(label2!=null) {
							label_sum[j]+=simu;
							if(label1.getValue()==label2.getValue()) {
								labelCon[j]+=simu;
							}
						}
					}
				}
				labelCon[j]/=label_sum[j];
			}
			
			double[] classPro=new double[numClass];
			for(int j=0;j<eLabelSet.getLabelSetSize(); j++) {
				classPro[eLabelSet.getLabel(j).getValue()]+=labelCon[j];
			}
			nclassMap.put(e.getId(), Utils.maxIndex(classPro));
			
		}

		for (int i = 0; i < numExample; i++) {
			Example example = dataset.getExampleByIndex(i);
			if(nclassMap.get(example.getId())!=example.getIntegratedLabel().getValue()) {
				noiseSet.addExample(example);
			}
			else {
				cleanSet.addExample(example);
			}
		}	

		//build two classifiers to correct the noise instances
		Classifier classifier1 = new J48();
		Classifier classifier2 = new NaiveBayes();
		classifier1.buildClassifier(cleanSet);
		classifier2.buildClassifier(cleanSet);
		for (int i = 0; i < noiseSet.getExampleSize(); i++) {
			Example e = noiseSet.getExampleByIndex(i);
			if ((int) classifier1.classifyInstance(e) == (int) classifier2.classifyInstance(e)
					&& (int) classifier1.classifyInstance(e) != nclassMap.get(e.getId())) {
				e.getIntegratedLabel().setValue((int) classifier1.classifyInstance(e));
				e.setTrainingLabel((int) classifier1.classifyInstance(e));
			}else {
				e.getIntegratedLabel().setValue(nclassMap.get(e.getId()));
				e.setTrainingLabel(nclassMap.get(e.getId()));
			}
			cleanSet.addExample(e);
		}
		return cleanSet;

	}

}
