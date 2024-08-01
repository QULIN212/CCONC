package ceka.LDNC;

import java.util.*;

import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Label;
import ceka.core.Worker;

import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.MultiNoisyLabelSet;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.core.*;

public class LDNC {

    public Dataset ldnc(Dataset dataset) throws Exception{
        //calculate label distribution
        HashMap<String, Double[]>label_dis = new HashMap<String, Double[]>();
        for(int i=0;i<dataset.getExampleSize();i++){
            Double[] dis = new Double[dataset.getCategorySize()];
            for(int n=0;n<dataset.getCategorySize();n++){
                dis[n]=0.0;
            }
            label_dis.put(dataset.getExampleByIndex(i).getId(),dis);
            MultiNoisyLabelSet exampleMultipleNoisyLabelSet=dataset.getExampleByIndex(i).getMultipleNoisyLabelSet(0);
            for(int j=0;j<exampleMultipleNoisyLabelSet.getLabelSetSize();j++){
                label_dis.get(dataset.getExampleByIndex(i).getId())[exampleMultipleNoisyLabelSet.getLabel(j).getValue()]++;
            }
            for(int k=0;k<dataset.getCategorySize();k++){
                label_dis.get(dataset.getExampleByIndex(i).getId())[dataset.getCategory(k).getValue()]/=exampleMultipleNoisyLabelSet.getLabelSetSize();
            }
        }

        // Initialize the clean set and noise set
        Dataset cleanSet = dataset.generateEmpty();
        Dataset noiseSet = dataset.generateEmpty();
        for(int i = 0;i < dataset.getCategorySize();i++) {
            Category cate = dataset.getCategory(i);
            cleanSet.addCategory(cate);
            noiseSet.addCategory(cate);
        }


        //divide dataset into clean set and noise set
        for(int i=0;i< dataset.getExampleSize();i++){
            HashMap<Integer,Double>ex_label_dis = new HashMap<Integer,Double>();
            for(int j=0;j<dataset.getCategorySize();j++){
                ex_label_dis.put(dataset.getCategory(j).getValue(),label_dis.get(dataset.getExampleByIndex(i).getId())[dataset.getCategory(j).getValue()]);
            }
            //sort
            List<Map.Entry<Integer, Double>> entryList = new ArrayList<>(ex_label_dis.entrySet());
            Collections.sort(entryList, (o1, o2) -> o2.getValue().compareTo(o1.getValue())); //降序
//            for (int k = 0; k < dataset.getCategorySize(); k++) { // 获取前两个元素
//                Map.Entry<Integer, Double> entry = entryList.get(k);
//                System.out.println("Key: " + entry.getKey() + ", Value: " + entry.getValue());
//            }
            Map.Entry<Integer, Double> entry0 = entryList.get(0);
            Map.Entry<Integer, Double> entry1 = entryList.get(1);
            if((entry0.getValue()-entry1.getValue()<=0.2)){
                noiseSet.addExample(dataset.getExampleByIndex(i));
            }
            else{
                cleanSet.addExample(dataset.getExampleByIndex(i));
            }
        }
//        System.out.println("cleanset:"+cleanSet.getExampleSize());
//        System.out.println("noiseset:"+noiseSet.getExampleSize());

        //correction
        Classifier classifier=new J48();
        classifier.buildClassifier(cleanSet);
        // Correct instances of noise set
        for(int i = 0;i < noiseSet.getExampleSize();i++) {
            Example e = noiseSet.getExampleByIndex(i);
            int label1 = (int)classifier.classifyInstance(e);
            e.getIntegratedLabel().setValue(label1);
            e.setTrainingLabel(label1);
            cleanSet.addExample(e);
        }
        return cleanSet;
    }
}

