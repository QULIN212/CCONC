package ceka.CCONC;

import java.util.*;

import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Worker;

import ceka.core.Example;
import ceka.core.MultiNoisyLabelSet;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.Filter;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.*;

public class CCONC {

    public Dataset copyDataset(Dataset dataset) throws Exception{
        Dataset newdataset = dataset.generateEmpty();
        for(int i = 0;i < dataset.getCategorySize();i++) {
            Category cate = dataset.getCategory(i);
            newdataset.addCategory(cate);
        }
        for(int i = 0;i < dataset.getExampleSize();i++) {
            Example e = dataset.getExampleByIndex(i);
            newdataset.addExample(e);

            for(int j = 0;j < e.getWorkerIdList().size();j++) {
                String wID = e.getWorkerIdList().get(j);
                Worker worker = newdataset.getWorkerById(wID);
                if(worker == null) {
                    newdataset.addWorker(worker = new Worker(wID));
                }
            }

        }
        return newdataset;
    }

    public Dataset cconc(Dataset dataset) throws Exception{

        //calculate worker's accuracy
        HashMap<String, Double>worker_acc = new HashMap<String, Double>();
        for(int i=0;i<dataset.getWorkerSize();i++){
            MultiNoisyLabelSet workerNoisyLabelSet=dataset.getWorkerByIndex(i).getMultipleNoisyLabelSet(0);
            int acc_count=0;
            for(int j=0;j<workerNoisyLabelSet.getLabelSetSize();j++){
                if(workerNoisyLabelSet.getLabel(j).getValue()==dataset.getExampleById(workerNoisyLabelSet.getLabel(j).getExampleId()).getIntegratedLabel().getValue()){
                    acc_count++;
                }
            }
//            System.out.println(acc_count);
            worker_acc.put(dataset.getWorkerByIndex(i).getId(),  ((double)acc_count/workerNoisyLabelSet.getLabelSetSize()));
        }

//        for(int i=0;i<dataset.getWorkerSize();i++){
//            System.out.println("worker"+dataset.getWorkerByIndex(i).getId()+"权重为"+worker_acc.get(dataset.getWorkerByIndex(i).getId()));
//        }


        //calculate class-confidence for each class in each instance
        HashMap<String,Double[]>class_conf = new HashMap<String,Double[]>();

        for(int i=0;i< dataset.getExampleSize();i++){
            //initialize double[]
            Double[] conf = new Double[dataset.getCategorySize()];
            for(int n=0;n<dataset.getCategorySize();n++){
                conf[n]=0.0;
            }
            class_conf.put(dataset.getExampleByIndex(i).getId(),conf);
            MultiNoisyLabelSet exampleMultipleNoisyLabelSet=dataset.getExampleByIndex(i).getMultipleNoisyLabelSet(0);
            double totalWorkerAcc=0.0;
            for(int j=0;j<exampleMultipleNoisyLabelSet.getLabelSetSize();j++){
                class_conf.get(dataset.getExampleByIndex(i).getId())[exampleMultipleNoisyLabelSet.getLabel(j).getValue()]+=worker_acc.get(exampleMultipleNoisyLabelSet.getLabel(j).getWorkerId());
                totalWorkerAcc+=worker_acc.get(exampleMultipleNoisyLabelSet.getLabel(j).getWorkerId());
            }
            for(int k=0;k<dataset.getCategorySize();k++){
                class_conf.get(dataset.getExampleByIndex(i).getId())[dataset.getCategory(k).getValue()]/=totalWorkerAcc;
//                System.out.println(class_conf.get(dataset.getExampleByIndex(i).getId())[dataset.getCategory(k).getValue()]);
            }
//            System.out.println(totalWorkerAcc);
        }



        //add class-confidence to attributes
        //add attribute list
        Dataset newdataset=copyDataset(dataset);
        ArrayList<Attribute> newAttributes = new ArrayList<>();
        for(int i=0;i< newdataset.getCategorySize();i++){
            newAttributes.add(new Attribute("newAttribute"+i));
        }
        //modify dataset structure
        int labelIndex = newdataset.classIndex(); // 获取标签属性的索引
//        System.out.println("Class index: " + newdataset.classIndex()); // 检查这是否输出 -1
        if (labelIndex == -1) {
            labelIndex = newdataset.numAttributes(); // 如果没有设置类别索引，则假设标签在最后
        }
        // 在标签属性之前插入新属性
        for (int i = 0; i < newAttributes.size(); i++) {
            newdataset.insertAttributeAt(newAttributes.get(i), labelIndex + i);
        }
        //更新每个实例
        for (int i = 0; i < newdataset.numInstances(); i++) {
            Instance instance = newdataset.instance(i);
            for(int j=0;j< newdataset.getCategorySize();j++){
                instance.setValue(labelIndex + j, class_conf.get(newdataset.getExampleByIndex(i).getId())[newdataset.getCategory(j).getValue()]);//为newAttribute设置值
            }
        }

//        System.out.println(newdataset.numAttributes());
//        System.out.println(newdataset.classIndex());

        // Initialize the clean set and noise set
        Dataset cleanSet = newdataset.generateEmpty();
        Dataset noiseSet = newdataset.generateEmpty();
        for(int i = 0;i < newdataset.getCategorySize();i++) {
            Category cate = newdataset.getCategory(i);
            cleanSet.addCategory(cate);
            noiseSet.addCategory(cate);
        }

        //divide dataset into clean set and noise set
        for(int i=0;i<newdataset.getExampleSize();i++){
            HashMap<Integer,Double>ex_class_conf = new HashMap<Integer,Double>();
            for(int j=0;j<newdataset.getCategorySize();j++){
                ex_class_conf.put(newdataset.getCategory(j).getValue(),class_conf.get(newdataset.getExampleByIndex(i).getId())[newdataset.getCategory(j).getValue()]);
            }
            //sort class-confidence
            List<Map.Entry<Integer, Double>> entryList = new ArrayList<>(ex_class_conf.entrySet());
            Collections.sort(entryList, (o1, o2) -> o2.getValue().compareTo(o1.getValue())); //降序

//            for (int k = 0; k < 2; k++) { // 获取前两个元素
//                Map.Entry<Integer, Double> entry = entryList.get(k);
//                System.out.println("Key: " + entry.getKey() + ", Value: " + entry.getValue());
//            }
            Map.Entry<Integer, Double> entry0 = entryList.get(0);
            Map.Entry<Integer, Double> entry1 = entryList.get(1);
            if((entry0.getValue()-entry1.getValue())<=0.2){
                noiseSet.addExample(newdataset.getExampleByIndex(i));
            }
            else{
                cleanSet.addExample(newdataset.getExampleByIndex(i));
            }
        }
        //entry0.getKey()!=newdataset.getExampleByIndex(i).getIntegratedLabel().getValue()&&(entry0.getValue()-entry1.getValue()<=0.2)
//        System.out.println(cleanSet.getExampleSize());
//        System.out.println(noiseSet.getExampleSize());

        //model training
        //smote Oversampling
        SMOTE smote=new SMOTE();
        smote.setInputFormat(cleanSet);
        smote.setPercentage(100.0); // 过采样100%

//        smote.setPercentage(50.0); // 过采样50%


        smote.setNearestNeighbors(5); // 5个最近邻
        Instances smoteData= Filter.useFilter(cleanSet,smote);
//        System.out.println(smoteData.numInstances());
        //AdaBoostM1
        AdaBoostM1 adaBoost = new AdaBoostM1();
        adaBoost.setClassifier(new J48());  // 设置J48作为AdaBoost的基分类器
        adaBoost.setNumIterations(100); // 设置迭代次数
//        adaBoost.setWeightThreshold(100);
        adaBoost.buildClassifier(smoteData);
        //NaiveBayes
        Classifier classifier2 = new NaiveBayes();
        classifier2.buildClassifier(smoteData);
        //IBK
        Classifier classifier3 = new IBk();
        classifier3.buildClassifier(smoteData);

        //predict
        for(int i = 0;i < noiseSet.getExampleSize();i++) {
            Example e = noiseSet.getExampleByIndex(i);
            int label1 = (int)adaBoost.classifyInstance(e);
            int label2 = (int)classifier2.classifyInstance(e);
            int label3 = (int)classifier3.classifyInstance(e);
            if(label1 == label2 && label1 == label3 && label2 == label3 && label1 != e.getIntegratedLabel().getValue()){
                e.getIntegratedLabel().setValue(label1);
                e.setTrainingLabel(label1);
                cleanSet.addExample(e);
            }

        }
//        System.out.println(cleanSet.getExampleSize());
        return cleanSet;
    }


}
