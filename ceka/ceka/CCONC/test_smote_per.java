package ceka.CCONC;
import ceka.consensus.MajorityVote;
import ceka.converters.FileLoader;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Worker;
import ceka.simulation.MockWorker;
import ceka.simulation.SingleQualLabelingStrategy;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;

public class test_smote_per{

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
        double meanNoiseRatio_CCONC=0.0;
        double meanNoiseRatio_CCONC100=0.0;
        double meanNoiseRatio_CCONC150=0.0;
        double meanNoiseRatio_CCONC200=0.0;

        double mean_runTime_CCONC=0;
        double mean_runTime_CCONC100=0;
        double mean_runTime_CCONC150=0;
        double mean_runTime_CCONC200=0;

        try {

            //模拟
            String resultPath="E:\\data\\result\\CCONC_result_smote_real.txt";
            FileOutputStream f=new FileOutputStream(new File(resultPath));
            PrintStream result=new PrintStream(f);
            result.format("%-20s %-10s %-10s %-10s %-10s", "Dataset","CCONC","CCONC100","CCONC150","CCONC200");
            result.println();



            for(int i=0;i<names.length;i++)
            {


                double noiseRatio_CCONC=0.0;
                double noiseRatio_CCONC100=0.0;
                double noiseRatio_CCONC150=0.0;
                double noiseRatio_CCONC200=0.0;


                double runTime_CCONC=0;
                double runTime_CCONC100=0;
                double runTime_CCONC150=0;
                double runTime_CCONC200=0;


                //模拟
                test_smote_per experiment=new test_smote_per();
                Dataset dataset=experiment.readRealDataset(i);

                for(int t=0;t<t_num;t++) {

                    //experiment.simulate(dataset, 7, 0.55,0.75);

                    double startTime=0;
                    double endTime=0;


                    //MV
                    MajorityVote mv=new MajorityVote();
                    mv.doInference(dataset);
                    System.out.println("time "+(t+1)+"\t 开始");

                    //mynew
                    startTime=System.currentTimeMillis();
                    CCONC mynew=new CCONC();
                    Dataset dataset_corrected_mynew = mynew.cconc(dataset);
                    endTime=System.currentTimeMillis();
                    noiseRatio_CCONC += experiment.getNoiseRatio(dataset_corrected_mynew);
                    runTime_CCONC+=(endTime-startTime)/1000.0;
                    System.out.println("CCONC Completed");

                    //100
                    startTime=System.currentTimeMillis();
                    CCONC100 cconc100=new CCONC100();
                    Dataset dataset_corrected_cconc100 = cconc100.cconc100(dataset);
                    endTime=System.currentTimeMillis();
                    noiseRatio_CCONC100 += experiment.getNoiseRatio(dataset_corrected_cconc100);
                    runTime_CCONC100+=(endTime-startTime)/1000.0;
                    System.out.println("CCONC100 Completed");

                    //150
                    startTime=System.currentTimeMillis();
                    CCONC150 cconc150=new CCONC150();
                    Dataset dataset_corrected_cconc150 = cconc150.cconc150(dataset);
                    endTime=System.currentTimeMillis();
                    noiseRatio_CCONC150 += experiment.getNoiseRatio(dataset_corrected_cconc150);
                    runTime_CCONC150+=(endTime-startTime)/1000.0;
                    System.out.println("CCONC150 Completed");

                    //200
                    startTime=System.currentTimeMillis();
                    CCONC200 cconc200=new CCONC200();
                    Dataset dataset_corrected_cconc200 = cconc200.cconc200(dataset);
                    endTime=System.currentTimeMillis();
                    noiseRatio_CCONC200 += experiment.getNoiseRatio(dataset_corrected_cconc200);
                    runTime_CCONC200+=(endTime-startTime)/1000.0;
                    System.out.println("CCONC200 Completed");

                    System.out.println("time "+(t+1)+"\t Completed!");
                }

                noiseRatio_CCONC/=t_num;
                noiseRatio_CCONC100/=t_num;
                noiseRatio_CCONC150/=t_num;
                noiseRatio_CCONC200/=t_num;

                meanNoiseRatio_CCONC+=noiseRatio_CCONC;
                meanNoiseRatio_CCONC100+=noiseRatio_CCONC100;
                meanNoiseRatio_CCONC150+=noiseRatio_CCONC150;
                meanNoiseRatio_CCONC200+=noiseRatio_CCONC200;



                runTime_CCONC/=t_num;
                runTime_CCONC100/=t_num;
                runTime_CCONC150/=t_num;
                runTime_CCONC200/=t_num;

                mean_runTime_CCONC+=runTime_CCONC;
                mean_runTime_CCONC100+=runTime_CCONC100;
                mean_runTime_CCONC150+=runTime_CCONC150;
                mean_runTime_CCONC200+=runTime_CCONC200;

                result.format("%-20s %-10.2f %-10.2f %-10.2f %-10.2f", names[i],noiseRatio_CCONC,noiseRatio_CCONC100,noiseRatio_CCONC150,noiseRatio_CCONC200);
                result.println();
                result_t.format("%-20s %-10.5f %-10.5f %-10.5f %-10.5f", names[i],runTime_CCONC,runTime_CCONC100,runTime_CCONC150,runTime_CCONC200);
                result_t.println();


                System.out.println(names[i]+" complete!");

            }
            meanNoiseRatio_CCONC/=names.length;
            meanNoiseRatio_CCONC100/=names.length;
            meanNoiseRatio_CCONC150/=names.length;
            meanNoiseRatio_CCONC200/=names.length;


            result.format("%-20s %-10.2f %-10.2f %-10.2f %-10.2f", "Mean",meanNoiseRatio_CCONC,meanNoiseRatio_CCONC100,meanNoiseRatio_CCONC150,meanNoiseRatio_CCONC200);
            result.println();
            result.close();

            mean_runTime_CCONC/=names.length;
            mean_runTime_CCONC100/=names.length;
            mean_runTime_CCONC150/=names.length;
            mean_runTime_CCONC200/=names.length;
            result_t.format("%-20s %-10.5f %-10.5f %-10.5f %-10.5f", "Mean",mean_runTime_CCONC,mean_runTime_CCONC100,mean_runTime_CCONC150,mean_runTime_CCONC200);
            result_t.println();
            result_t.close();


            System.out.println("Complete!!!");

        }catch(Exception e)
        {
            System.out.println(e);
        }
    }

}
