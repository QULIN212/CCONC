package ceka.noise.clustering;

import ceka.core.Dataset;
import ceka.core.Example;

import java.util.ArrayList;

import weka.clusterers.Clusterer;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author jian
 */
public class ClusterCorrection 
{
    private Dataset classDataset;
    private Dataset classlessDataset;
    private Clusterer[] clusterers;
    public ClusterCorrection(Dataset dataset, String tempPath, Clusterer[] clusterers)
    {
    	try
    	{
            this.clusterers = clusterers;
            classDataset = dataset.generateEmpty();
            classlessDataset = dataset.generateEmpty();
            for(int i = 0; i < dataset.getExampleSize(); i++)
            {
            	classDataset.addExample(dataset.getExampleByIndex(i));
            	classlessDataset.addExample(dataset.getExampleByIndex(i));
            }
            for(int i = 0; i < dataset.getCategorySize(); i++)
            {
            	classDataset.addCategory(dataset.getCategory(i));
            	classlessDataset.addCategory(dataset.getCategory(i));
            }
            for(int i = 0; i < classlessDataset.getExampleSize(); i++)
            {
            	classlessDataset.getExampleByIndex(i).setClassValue(0);
            	classlessDataset.getExampleByIndex(i).getTrueLabel().setValue(0);
            	classlessDataset.getExampleByIndex(i).getIntegratedLabel().setValue(0);
            }
            classlessDataset.setClassIndex(-1);
            Normalize normalize = new Normalize();
            normalize.setInputFormat(classlessDataset);
            normalize.useFilter(classlessDataset, normalize);
    	}
    	catch(Exception e)
    	{
    		e.printStackTrace();
    		System.exit(1);
    	}
    }


    public Dataset correction() throws Exception
    {
        int numClusteringAlgs = clusterers.length;
        int numClasses = classDataset.getCategorySize();
        double[][][] distributions = new double[numClusteringAlgs][numClasses][classDataset.getExampleSize()];
        int[][] memberships = new int[numClusteringAlgs][classDataset.getExampleSize()];
        double[] overallDist = new double[numClasses];
        for(int i = 0; i < classDataset.getExampleSize(); i++)
        {
            overallDist[(int)classDataset.getExampleByIndex(i).getIntegratedLabel().getValue()]++;
        }
        overallDist = normalize(overallDist);
        for(int i = 0; i < numClusteringAlgs; i++)
        {
            //System.out.println("On clustering algorithm " + (i + 1));
            clusterers[i].buildClusterer(classlessDataset);
            for(int j = 0; j < classlessDataset.getExampleSize(); j++)
            {
                memberships[i][j] = clusterers[i].clusterInstance(classlessDataset.getExampleByIndex(j));
            }
        }

        for(int i = 0; i < numClusteringAlgs; i++)
        {
            //System.out.println("For clustering algorithm " + (i + 1));
            for(int j = 0; j < clusterers[i].numberOfClusters(); j++)
            {
                //System.out.println("For cluster " + (j + 1));
                ArrayList<Integer> indices = new ArrayList();
                for(int k = 0; k < classlessDataset.getExampleSize(); k++)
                {
                    if(memberships[i][k] == j)
                       indices.add(k);
                }
                
                if(indices.isEmpty())
                    continue;
                
                double[] classDists = new double[numClasses];
                
                for(int n = 0; n < numClasses; n++)
                {
                    classDists[n] = 0;
                }
                
                for(Integer integer : indices)
                {
                    classDists[classDataset.getExampleByIndex(integer).getIntegratedLabel().getValue()]++;
                }
                
                for(int n = 0; n < numClasses; n++)
                {
                    for(Integer integer : indices)
                    {
                        distributions[i][n][integer] = classDists[n];
                    }
                } 
            }
        }
        
        for(int i = 0; i < classDataset.getExampleSize(); i++)
        {
            double maxDist = Double.NEGATIVE_INFINITY;
            int maxIndex = -1;
            for(int j = 0; j < numClasses; j++)
            {
                double sum = 0;
                for(int k = 0; k < numClusteringAlgs; k++)
                {
                    double total = 0;
                    double[] instanceDist = new double[numClasses];
                    for(int l = 0; l < numClasses; l++)
                    {
                        instanceDist[l] = distributions[k][l][i];
                    }
                    sum += computeWeight(overallDist,instanceDist)[j];
                }
                if(sum > maxDist)
                {
                    maxDist = sum;
                    maxIndex = j;
                }
            }
            Example e = classDataset.getExampleByIndex(i);
            if(maxDist > 0)
            {
                e.getIntegratedLabel().setValue(maxIndex);
                try
                {
                    e.setClassValue("" + maxIndex);
                }
                catch(weka.core.UnassignedClassException uce)
                {
                    e.setValue(e.numAttributes() - 1, "" + maxIndex);
                }
            }
        }
        return classDataset;
    }
    
     private double[] computeWeight(double[] standard, double[] instance)
     {
        int numInstances = (int)sum(instance);
        instance = normalize(instance);
        double[] result = new double[standard.length];
        double multiplier = Math.log10(numInstances);
        if(multiplier > 2)
            multiplier = 2;
        for(int i = 0; i < standard.length; i++)
        {
            //result[i] = (instance[i] - standard[i]) / standard[i]
            result[i] = (instance[i] - (1.0/(double)standard.length)) / standard[i];
            result[i] *= multiplier;
        }
        
        return result;
    }
     
     private double sum(double[] values)
     {
         double total = 0;
         for(int i = 0; i < values.length; i++)
         {
             total += values[i];
         }
         return total;
     }
     
     private double[] normalize(double[] values)
     {
         double total = sum(values);
         for(int i = 0; i < values.length; i++)
         {
             values[i] /= total;
         }
         return values;
     }
}


