package com.example.jadhosn.group9_a3;

import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;
import java.util.Vector;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

public class Train extends AppCompatActivity {

    //Libsvm code - literally: libsvm/java/svm_train.java
    private svm_parameter params;
    private svm_problem prob;
    private svm_model model;

    private int cross_validation;
    private int nr_fold;
    private double acc =0;


    public void parameters()
    {

        params = new svm_parameter();
        params.svm_type = svm_parameter.C_SVC;
        params.kernel_type = svm_parameter.POLY;
        params.degree = 3;
        params.gamma = 0.008;
        params.coef0 = 0;
        params.nu = 0.5;
        params.cache_size = 100;
        params.C = 10000;
        params.eps = 1e-3;
        params.p = 0.1;
        params.shrinking = 1;
        params.probability = 0;
        params.nr_weight = 0;
        params.weight_label = new int[0];
        params.weight = new double[0];
        cross_validation = 1;
        nr_fold = 3;//must be greater than 2
    }

    public void set_data() throws IOException
    {
        //read our converted dataset into libsvm format
        FileInputStream fis = new FileInputStream(new File(Environment.getExternalStorageDirectory().toString() + "/Android/Data/CSE535_ASSIGNMENT3/group9.txt"));
        InputStreamReader isr = new InputStreamReader(fis);
        BufferedReader br = new BufferedReader(isr);

        Vector<Double> vy = new Vector<Double>();
        Vector<svm_node[]> vx = new Vector<svm_node[]>();
        int midx = 0;

        while(true)
        {
            String line = br.readLine();//read line by line from buffer reader
            if(line == null) break;//end of file

            StringTokenizer st = new StringTokenizer(line," \t\n\r\f:");

            vy.addElement(Double.valueOf(st.nextToken()).doubleValue());

            int m = st.countTokens()/2;

            svm_node[] x = new svm_node[m];
            for(int j=0;j<m;j++)
            {
                x[j] = new svm_node();
                x[j].index = Integer.parseInt(st.nextToken());
                x[j].value = Double.valueOf(st.nextToken()).doubleValue();
            }
            if(m>0) midx = Math.max(midx, x[m-1].index);
            vx.addElement(x);
        }

        prob = new svm_problem();
        prob.l = vy.size();
        prob.x = new svm_node[prob.l][];

        for(int i = 0; i< prob.l; i++)
            prob.x[i] = vx.elementAt(i);

        prob.y = new double[prob.l];

        for(int i = 0; i< prob.l; i++)
            prob.y[i] = vy.elementAt(i);


        br.close();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_train);

        TextView display = (TextView) findViewById(R.id.display);
        TextView display2 = (TextView)findViewById(R.id.display2);

        parameters();



        display.setText("SVM Classifier:\n\n" +
                "kernel_type = "+params.kernel_type+"\n" +
                "degree = "+params.degree+"\n" +
                "gamma = "+params.gamma+"\n" +
                "cache_size = "+params.cache_size+"\n" +
                "C = "+params.C+"\n" +
                "nr_fold = "+nr_fold+"\n");


        try {
            String exception = "";
            set_data();
            exception = svm.svm_check_parameter(prob, params);
            if(exception != null)
                Toast.makeText(getBaseContext(), exception, Toast.LENGTH_LONG).show();
            if(cross_validation != 0) {cv();}
            else {model = svm.svm_train(prob, params);}
            display2.setText("SVM Accuracy is: "+ acc+" %");
        }
        catch(Exception ex) {
            //oups
        }
    }

    private void cv()
    {
        int i;
        int total_correct = 0;
        double[] target = new double[prob.l];

        svm.svm_cross_validation(prob, params,nr_fold,target);
        total_correct = 0;

        for(i=0; i< prob.l; i++)
            if(target[i] == prob.y[i])
                ++total_correct;
        acc = 100.0*total_correct/ prob.l;

    }
}
