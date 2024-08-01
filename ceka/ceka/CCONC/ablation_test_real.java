package ceka.CCONC;

import ceka.core.*;
import ceka.converters.FileLoader;
import ceka.consensus.MajorityVote;

import ceka.simulation.MockWorker;
import ceka.simulation.SingleQualLabelingStrategy;

import java.io.FileOutputStream;
import java.io.File;
import java.io.PrintStream;

public class ablation_test_real {

    private static int t_num=10;

    //真实世界数据集
    private static String[] names = {"leaves6","income94L10","labelme"};
    public Dataset readRealDataset(int m_choose) throws Exception{
        String dataDir = "E:\\data\\real-world\\CCONC\\";
        String arffXPath = dataDir + names[m_choose] + "\\" + names[m_choose] + ".arffx";
        String arffPath = dataDir + names[m_choose] + "\\" + names[m_choose] + ".arff";
        String goldPath = dataDir + names[m_choose] + "\\" + names[m_choose] + ".gold.txt";
        String responsePath = dataDir + names[m_choose] + "\\" + names[m_choose] + ".response.txt";

        Dataset dataset;
        try {
            dataset = FileLoader.loadFileX(responsePath, goldPath, arffXPath);
        } catch (Exception e) {
            dataset = FileLoader.loadFile(responsePath, goldPath, arffPath);
        }

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
        double meanNoiseRatio_CCONC=0.0;
        double meanNoiseRatio_CCONC_I=0.0;
        double meanNoiseRatio_CCONC_II=0.0;

        try {

            //模拟
            String resultPath="E:\\data\\result\\CCONC_result_ablation_real.txt";
            FileOutputStream f=new FileOutputStream(new File(resultPath));
            PrintStream result=new PrintStream(f);
            result.format("%-20s %-10s %-10s %-10s %10s", "Dataset","MV","CCONC","CCONC_I","CCONC_II");
            result.println();



            for(int i=0;i<names.length;i++)
            {


                double noiseRatio_MV=0.0;
                double noiseRatio_CCONC=0.0;
                double noiseRatio_CCONC_I=0.0;
                double noiseRatio_CCONC_II=0.0;



                ablation_test_real experiment=new ablation_test_real();
                Dataset dataset=experiment.readRealDataset(i);

                for(int t=0;t<t_num;t++) {

                    //MV
                    MajorityVote mv=new MajorityVote();
                    mv.doInference(dataset);
                    noiseRatio_MV+=experiment.getNoiseRatio(dataset);
                    System.out.println("time "+(t+1)+"\t 开始");

                    //mynew
                    CCONC mynew=new CCONC();
                    Dataset dataset_corrected_cconc = mynew.cconc(dataset);
                    noiseRatio_CCONC += experiment.getNoiseRatio(dataset_corrected_cconc);
                    System.out.println("CCONC Completed");

                    //CCONC_I
                    CCONC_I cconc_I=new CCONC_I();
                    Dataset dataset_corrected_cconc_I = cconc_I.cconc_I(dataset);
                    noiseRatio_CCONC_I += experiment.getNoiseRatio(dataset_corrected_cconc_I);
                    System.out.println("CCONC-I Completed");

                    //CCONC_II
                    CCONC_II cconc_II=new CCONC_II();
                    Dataset dataset_corrected_cconc_II = cconc_II.cconc_II(dataset);
                    noiseRatio_CCONC_II += experiment.getNoiseRatio(dataset_corrected_cconc_II);
                    System.out.println("CCONC-II Completed");

                    System.out.println("time "+(t+1)+"\t Completed!");
                }

                noiseRatio_MV/=t_num;
                noiseRatio_CCONC/=t_num;
                noiseRatio_CCONC_I/=t_num;
                noiseRatio_CCONC_II/=t_num;

                meanNoiseRatio_MV+=noiseRatio_MV;
                meanNoiseRatio_CCONC+=noiseRatio_CCONC;
                meanNoiseRatio_CCONC_I+=noiseRatio_CCONC_I;
                meanNoiseRatio_CCONC_II+=noiseRatio_CCONC_II;

                result.format("%-20s %-10.2f %-10.2f %-10.2f %-10.2f", names[i],noiseRatio_MV,noiseRatio_CCONC,noiseRatio_CCONC_I,noiseRatio_CCONC_II);
                result.println();


                System.out.println(names[i]+" complete!");

            }
            meanNoiseRatio_MV/=names.length;
            meanNoiseRatio_CCONC/=names.length;
            meanNoiseRatio_CCONC_I/=names.length;
            meanNoiseRatio_CCONC_II/=names.length;

            result.format("%-20s %-10.2f %-10.2f %-10.2f %-10.2f", "Mean",meanNoiseRatio_MV,meanNoiseRatio_CCONC,meanNoiseRatio_CCONC_I,meanNoiseRatio_CCONC_II);
            result.println();
            result.close();


            System.out.println("Complete!!!");

        }catch(Exception e)
        {
            System.out.println(e);
        }
    }
}
