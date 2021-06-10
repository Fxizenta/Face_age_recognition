package com.smart;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class pyjava {
    public static void main(String[] args) {
        // TODO Auto-generated method stub
        String path = " D:\\deepface\\age-gender-estimation\\demo.py --image_dir";
        String image_path = " D:\\deepface\\age-gender-estimation\\data\\imdb_crop\\test";
        try {
            String args1 = "python"+path+image_path;
            Process proc = Runtime.getRuntime().exec(args1);// 执行py文件

            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = null;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            proc.waitFor();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        //System.out.println(args);
    }
}

