package ceka.CCONC;

import ceka.CWVNC.CWVNC;
import ceka.LDNC.LDNC;
import ceka.NWVNC.NWVNC;
import ceka.WSNC.WSNC;
import ceka.core.*;
import ceka.converters.FileLoader;
import ceka.converters.FileSaver;
import ceka.consensus.MajorityVote;

import ceka.noise.ClassificationFilter;

import ceka.noise.PolishingLabels;
import ceka.noise.SelfTrainCorrection;
import ceka.noise.avnc.AdaptiveClassificationFilter;
import ceka.noise.avnc.VoteCorrection;
import ceka.noise.avnc.WorkerStat;
import ceka.noise.clustering.ClusterCorrection;

import ceka.simulation.MockWorker;
import ceka.simulation.SingleQualLabelingStrategy;
import ceka.utils.DatasetManipulator;
import ceka.utils.Stochastics;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;

import java.io.FileOutputStream;
import java.io.File;
import java.io.PrintStream;

public class test_cconc {

    private static int t_num=10;
    //模拟
    private static String[] names= {"anneal","balance-scale","biodeg","breast-cancer","breast-w","car","credit-a","credit-g","diabetes","heart-c","heart-h","heart-statlog","hepatitis","horse-colic","ionosphere","iris","kr-vs-kp","labor","mushroom","segment","sick","sonar","spambase","tic-tac-toe","vehicle","vote","vowel","waveform","letter"};

    public static Dataset readDataset(int m_choose) throws Exception {
		String arffPath = "E:\\data\\synthetic\\" + names[m_choose] + "\\" + names[m_choose]
				+ ".arff";
		Dataset dataset = FileLoader.loadFile(arffPath);

		return dataset;
	}

    public static void simulate(Dataset dataset,int numOfWorkers,double lowQuality,double highQuality)
    {

        MockWorker[] mockworkers=new MockWorker[numOfWorkers];
        for(int i=0;i<numOfWorkers;i++) {
            double quality=randdouble(lowQuality, highQuality);
            SingleQualLabelingStrategy strategy=new SingleQualLabelingStrategy(quality);
            mockworkers[i]=new MockWorker(new Integer(i).toString());
            mockworkers[i].setSingleQuality(quality);
            mockworkers[i].labeling(dataset, strategy);
        }

    }
    public static double randdouble(double max, double min) {
        return (Math.random() * (max - min) + min) ;
    }

    public double getNoiseRatio(Dataset dataset) throws Exception{
        int count = 0;
        for(int i = 0;i < dataset.getExampleSize();i++) {
            if(dataset.getExampleByIndex(i).getIntegratedLabel().getValue() != dataset.getExampleByIndex(i).getTrueLabel().getValue()) {
                count++;
            }
        }
        return 100*(double)count/dataset.getExampleSize();
    }

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

    public static void main(String[] args){
        double meanNoiseRatio_MV=0.0;
        double meanNoiseRatio_PL=0.0;
        double meanNoiseRatio_STC=0.0;
        double meanNoiseRatio_CC=0.0;
        double meanNoiseRatio_AVNC=0.0;
        double meanNoiseRatio_LDNC=0.0;
        double meanNoiseRatio_NWVNC=0.0;
        double meanNoiseRatio_WSNC=0.0;
        double meanNoiseRatio_CWVNC=0.0;
        double meanNoiseRatio_new=0.0;

        double mean_runTime_MV=0;
        double mean_runTime_PL=0;
        double mean_runTime_STC=0;
        double mean_runTime_CC=0;
        double mean_runTime_AVNC=0;
        double mean_runTime_LDNC=0;
        double mean_runTime_NWVNC=0.0;
        double mean_runTime_WSNC=0.0;
        double mean_runTime_CWVNC=0.0;
        double mean_runTime_new=0;

        try {

            //模拟
            String resultPath="E:\\data\\result\\CCONC_result.txt";
            FileOutputStream f=new FileOutputStream(new File(resultPath));
            PrintStream result=new PrintStream(f);
            result.format("%-20s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s", "Dataset","MV","PL","STC","CC","AVNC","LDNC","NWVNC","WSNC","CWVNC","CCONC");
            result.println();
            //时间
            String resultPath_t="E:\\data\\result\\CCONC_result_t.txt";
            FileOutputStream f_t=new FileOutputStream(new File(resultPath_t));
            PrintStream result_t=new PrintStream(f_t);
            result_t.format("%-20s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s", "Dataset","MV","PL","STC","CC","AVNC","LDNC","NWVNC","WSNC","CWVNC","CCONC");
            result_t.println();



            for(int i=0;i<names.length;i++)
            {


                double noiseRatio_MV=0.0;
                double noiseRatio_PL=0.0;
                double noiseRatio_STC=0.0;
                double noiseRatio_CC=0.0;
                double noiseRatio_AVNC=0.0;
                double noiseRatio_LDNC=0.0;
				double noiseRatio_NWVNC=0.0;
				double noiseRatio_WSNC=0.0;
                double noiseRatio_CWVNC=0.0;
                double noiseRatio_new=0.0;

                double runTime_MV=0;
                double runTime_PL=0;
                double runTime_STC=0;
                double runTime_CC=0;
                double runTime_AVNC=0;
                double runTime_LDNC=0;
                double runTime_NWVNC=0.0;
				double runTime_WSNC=0.0;
                double runTime_CWVNC=0.0;
                double runTime_new=0;


                //模拟
                test_cconc experiment=new test_cconc();
                Dataset dataset=experiment.readDataset(i);

                for(int t=0;t<t_num;t++) {

                    experiment.simulate(dataset, 7, 0.55,0.75);

                    double startTime=0;
                    double endTime=0;


                    //MV
                    startTime=System.currentTimeMillis();
                    MajorityVote mv=new MajorityVote();
                    mv.doInference(dataset);
                    endTime=System.currentTimeMillis();
                    noiseRatio_MV+=experiment.getNoiseRatio(dataset);
                    runTime_MV+=(endTime-startTime)/1000.0;
                    System.out.println("time "+(t+1)+"\t 开始");

                    //PL
                    Dataset tempDataset1=experiment.copyDataset(dataset);
                    startTime=System.currentTimeMillis();
                    Classifier classifier1=new J48();
                    PolishingLabels pl=new PolishingLabels(classifier1);
                    Dataset dataset_corrected_pl=pl.polishLabels(tempDataset1);
                    endTime=System.currentTimeMillis();
                    noiseRatio_PL+=experiment.getNoiseRatio(dataset_corrected_pl);
                    runTime_PL+=(endTime-startTime)/1000.0;
                    System.out.println("PL Completed!");


                    //STC
                    Dataset tempDataset2=experiment.copyDataset(dataset);
                    startTime=System.currentTimeMillis();
                    Classifier[] classifiers1=new Classifier[1];
                    classifiers1[0]=new J48();
                    ClassificationFilter cf1=new ClassificationFilter(10);
                    cf1.filterNoise(tempDataset2, classifiers1);
                    Dataset cleanSet1=cf1.getCleansedDataset();
                    Dataset noiseSet1=cf1.getNoiseDataset();

                    Dataset[] tempdata=new Dataset[2];
                    SelfTrainCorrection stc=new SelfTrainCorrection(cleanSet1,noiseSet1,0.8);
                    Classifier classifier2=new J48();
                    tempdata=stc.correction(classifier2);
                    DatasetManipulator.addAllExamples(tempdata[0], tempdata[1]);
                    endTime=System.currentTimeMillis();
                    noiseRatio_STC+=experiment.getNoiseRatio(tempdata[0]);
                    runTime_STC+=(endTime-startTime)/1000.0;
                    System.out.println("STC Completed!");

                    //CC
                    Dataset tempDataset3=experiment.copyDataset(dataset);
                    startTime=System.currentTimeMillis();
                    int numClusters=10;
                    Clusterer[] clusterers=new Clusterer[numClusters];
                    for(int c=0;c<numClusters;c++)
                    {
                        int k=(int)(((double)(c+1)/(double)(numClusters))*(tempDataset3.getExampleSize()/2.0));
                        if(c==0) k=k+2;
                        SimpleKMeans simpleKMeans=new SimpleKMeans();
                        simpleKMeans.setMaxIterations(200);
                        simpleKMeans.setNumClusters(k);
                        clusterers[c]=simpleKMeans;
                    }

                    String tempPath="E:/data/output/output_CCONC/"+names[i]+t+".arff";
                    FileSaver.saveDatasetArff(tempDataset3, tempPath);
                    ClusterCorrection cc=new ClusterCorrection(tempDataset3,tempPath,clusterers);
                    Dataset dataset_corrected_cc=cc.correction();
                    endTime=System.currentTimeMillis();
                    noiseRatio_CC+=experiment.getNoiseRatio(dataset_corrected_cc);
                    runTime_CC+=(endTime-startTime)/1000.0;
                    System.out.println("CC Completed!");


                    //AVNC
                    Dataset tempDataset4=experiment.copyDataset(dataset);
                    startTime=System.currentTimeMillis();
                    WorkerStat workerStat=new WorkerStat();
                    double estimatedMeanProb=workerStat.calculateEstimatedMeanAcc(dataset);
                    double integratedCorrectProb=Stochastics.binomialIntegration(9, estimatedMeanProb);
                    int nfold=10;
                    int nModel=5;
                    AdaptiveClassificationFilter acf=new AdaptiveClassificationFilter(nfold,nModel);
                    acf.setMinEstimatedNoiseProportion(1-integratedCorrectProb);
                    acf.setMaxEstimatedNoiseProportion(1-estimatedMeanProb);

                    Classifier[] classifiers2=new Classifier[5];
                    for(int k=0;k<5;k++)
                        classifiers2[k]=new J48();
                    acf.filterNoise(tempDataset4, classifiers2);
                    Dataset cleanSet2=acf.getCleansedDataset();
                    Dataset noiseSet2=acf.getNoiseDataset();
                    Dataset[] highDatasets=acf.getHighQualityDatasets();

                    VoteCorrection corrector=new VoteCorrection();
                    corrector.correct(noiseSet2, highDatasets, classifiers2, (int)(highDatasets.length*0.5));
                    for(int k=0;k<noiseSet2.getExampleSize();k++)
                        cleanSet2.addExample(noiseSet2.getExampleByIndex(k));
                    endTime=System.currentTimeMillis();
                    noiseRatio_AVNC+=experiment.getNoiseRatio(cleanSet2);
                    runTime_AVNC+=(endTime-startTime)/1000.0;
                    System.out.println("AVNC Completed!");

                    //LDNC
                    Dataset tempDataset5=experiment.copyDataset(dataset);
                    startTime=System.currentTimeMillis();
                    LDNC ldnc=new LDNC();
                    Dataset dataset_corrected_ldnc=ldnc.ldnc(tempDataset5);
                    endTime=System.currentTimeMillis();
                    runTime_LDNC+=(endTime-startTime)/1000.0;
                    noiseRatio_LDNC+=experiment.getNoiseRatio(dataset_corrected_ldnc);
                    System.out.println("LDNC Completed!");


                    //NWVNC
                    Dataset tempDataset6=experiment.copyDataset(dataset);
                    startTime=System.currentTimeMillis();
                    NWVNC nwvnc = new NWVNC();
                    Dataset dataset_corrected_nwvnc = nwvnc.nwvnc(tempDataset6);
                    endTime=System.currentTimeMillis();
                    runTime_NWVNC+=(endTime-startTime)/1000.0;
                    noiseRatio_NWVNC += experiment.getNoiseRatio(dataset_corrected_nwvnc);
                    System.out.println("NWVNC Completed");

                    //WSNC
                    Dataset tempDataset7=experiment.copyDataset(dataset);
                    startTime=System.currentTimeMillis();
                    WSNC wsnc=new WSNC();
                    Dataset dataset_corrected_wsnc = wsnc.WSNC(tempDataset7,5);
                    endTime=System.currentTimeMillis();
                    runTime_WSNC+=(endTime-startTime)/1000.0;
                    noiseRatio_WSNC+=experiment.getNoiseRatio(dataset_corrected_wsnc);
                    System.out.println("WSNC Completed!");

                    // CWVNC
                    Dataset tempDataset8=experiment.copyDataset(dataset);
                    startTime=System.currentTimeMillis();
                    CWVNC cwvnc = new CWVNC();
                    Dataset dataset_corrected_cwvnc = cwvnc.CWVNC(tempDataset8);
                    endTime=System.currentTimeMillis();
                    runTime_CWVNC+=(endTime-startTime)/1000.0;
                    noiseRatio_CWVNC += experiment.getNoiseRatio(dataset_corrected_cwvnc);
                    System.out.println("CWVNC Completed");

                    //mynew
                    startTime=System.currentTimeMillis();
                    CCONC mynew=new CCONC();
                    Dataset dataset_corrected_mynew = mynew.cconc(dataset);
                    endTime=System.currentTimeMillis();
                    noiseRatio_new += experiment.getNoiseRatio(dataset_corrected_mynew);
                    runTime_new+=(endTime-startTime)/1000.0;
                    System.out.println("CCONC Completed");

                    System.out.println("time "+(t+1)+"\t Completed!");
                }

                noiseRatio_MV/=t_num;
                noiseRatio_PL/=t_num;
                noiseRatio_STC/=t_num;
                noiseRatio_CC/=t_num;
                noiseRatio_AVNC/=t_num;
                noiseRatio_LDNC/=t_num;
				noiseRatio_NWVNC/=t_num;
				noiseRatio_WSNC/=t_num;
                noiseRatio_CWVNC/=t_num;
                noiseRatio_new/=t_num;

                meanNoiseRatio_MV+=noiseRatio_MV;
                meanNoiseRatio_PL+=noiseRatio_PL;
                meanNoiseRatio_STC+=noiseRatio_STC;
                meanNoiseRatio_CC+=noiseRatio_CC;
                meanNoiseRatio_AVNC+=noiseRatio_AVNC;
                meanNoiseRatio_LDNC+=noiseRatio_LDNC;
				meanNoiseRatio_NWVNC+=noiseRatio_NWVNC;
				meanNoiseRatio_WSNC+=noiseRatio_WSNC;
                meanNoiseRatio_CWVNC+=noiseRatio_CWVNC;
                meanNoiseRatio_new+=noiseRatio_new;


                runTime_MV/=t_num;
                runTime_PL/=t_num;
                runTime_STC/=t_num;
                runTime_CC/=t_num;
                runTime_AVNC/=t_num;
                runTime_LDNC/=t_num;
				runTime_NWVNC/=t_num;
				runTime_WSNC/=t_num;
                runTime_CWVNC/=t_num;
                runTime_new/=t_num;

                mean_runTime_MV+=runTime_MV;
                mean_runTime_PL+=runTime_PL;
                mean_runTime_STC+=runTime_STC;
                mean_runTime_CC+=runTime_CC;
                mean_runTime_AVNC+=runTime_AVNC;
                mean_runTime_LDNC+=runTime_LDNC;
                mean_runTime_NWVNC+=runTime_NWVNC;
                mean_runTime_WSNC+=runTime_WSNC;
                mean_runTime_CWVNC+=runTime_CWVNC;
                mean_runTime_new+=runTime_new;

                result.format("%-20s %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f", names[i],noiseRatio_MV,noiseRatio_PL,noiseRatio_STC,noiseRatio_CC,noiseRatio_AVNC,noiseRatio_LDNC,noiseRatio_NWVNC,noiseRatio_WSNC,noiseRatio_CWVNC,noiseRatio_new);
                result.println();
                result_t.format("%-20s %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f", names[i],runTime_MV,runTime_PL,runTime_STC,runTime_CC,runTime_AVNC,runTime_LDNC,runTime_NWVNC,runTime_WSNC,runTime_CWVNC,runTime_new);
                result_t.println();


                System.out.println(names[i]+" complete!");

            }
            meanNoiseRatio_MV/=names.length;
            meanNoiseRatio_PL/=names.length;
            meanNoiseRatio_STC/=names.length;
            meanNoiseRatio_CC/=names.length;
            meanNoiseRatio_AVNC/=names.length;
            meanNoiseRatio_LDNC/=names.length;
            meanNoiseRatio_NWVNC/=names.length;
			meanNoiseRatio_WSNC/=names.length;
            meanNoiseRatio_CWVNC/=names.length;
            meanNoiseRatio_new/=names.length;

            result.format("%-20s %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f", "Mean",meanNoiseRatio_MV,meanNoiseRatio_PL,meanNoiseRatio_STC,meanNoiseRatio_CC,meanNoiseRatio_AVNC,meanNoiseRatio_LDNC,meanNoiseRatio_NWVNC,meanNoiseRatio_WSNC,meanNoiseRatio_CWVNC,meanNoiseRatio_new);
            result.println();
            result.close();

            mean_runTime_MV/=names.length;
            mean_runTime_PL/=names.length;
            mean_runTime_STC/=names.length;
            mean_runTime_CC/=names.length;
            mean_runTime_AVNC/=names.length;
            mean_runTime_LDNC/=names.length;
            mean_runTime_NWVNC/=names.length;
            mean_runTime_WSNC/=names.length;
            mean_runTime_CWVNC/=names.length;
            mean_runTime_new/=names.length;
            result_t.format("%-20s %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f", "Mean",mean_runTime_MV,mean_runTime_PL,mean_runTime_STC,mean_runTime_CC,mean_runTime_AVNC,mean_runTime_LDNC,mean_runTime_NWVNC,mean_runTime_WSNC,mean_runTime_CWVNC,mean_runTime_new);
            result_t.println();
            result_t.close();


            System.out.println("Complete!!!");

        }catch(Exception e)
        {
            System.out.println(e);
        }
    }
}
