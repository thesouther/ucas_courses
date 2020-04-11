import java.io.*;
import java.net.URI;
import java.util.List;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.io.IOException;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;

import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.MasterNotRunningException;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.ZooKeeperConnectionException;
import org.apache.hadoop.hbase.client.HBaseAdmin;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Put;

import org.apache.log4j.*;

/**
 *complie Hw1Grp0.java:
 *  javac Hw1Grp0.java 
 *run code like:
 *  java Hw1Grp0 R=/hw1/supplier.tbl S=/hw1/nation.tbl join:R2=S3 res:R4,S5,R6
 *
 * I set HDFS_BASE_PATH= "hdfs://localhost:9000". If you change the port of Hadoop,
 * please change this parameter.
 *
 * @author CongCong, 201928015029010, congcong19@mails.ucas.ac.cn
 * @since 22 March 2020
 */
public class Hw1Grp0 {
    private static final String TABLE_NAME = "Result";
	private static final String COLUME_FAMILY = "res";
    private static final String HDFS_BASE_PATH= "hdfs://localhost:9000";
    private static String hdfsFileR;
    private static String hdfsFileS;
    private static int hashJoinKeyR;
    private static int hashJoinKeyS;
    private static Integer[] selectListResR;
    private static Integer[] selectListResS;
    
    private static List<String[]> result = new ArrayList<String[]>();

    /**
	 * Read HDFS file anda return.
	 * @param fileName the specified file path.
	 * @return BufferedReader.
	 * @exception IOException On input error.
	 * @exception URISyntaxException throws when string could not be parsed as a URI reference.
	 * @see IOException
	 * @see URISyntaxException
	 */
    public BufferedReader HDFSReader(String fileName) throws IOException, URISyntaxException{
        // load HDFS file into inputstream
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(fileName), conf);
        Path path = new Path(fileName);
        FSDataInputStream in_stream = fs.open(path);
        BufferedReader in = new BufferedReader(new InputStreamReader(in_stream));
        // in.close();
        // fs.close();
        return in;
    }

    /**
	 * hash join process.
	 * @param fileName the specified file path.
     * @param fileR HDFS file R name.
     * @param fileS HDFS file S name.
     * @param joinKeyR join key R.
     * @param joinKeyS join key S.
     * @param listResR list of index of selected columns from file R.
     * @param listResS list of index of selected columns from file S.
	 * @exception IOException On input error.
	 * @exception URISyntaxException throws when string could not be parsed as a URI reference.
	 * @see IOException
	 * @see URISyntaxException
	 */
    public void processHashJoin(String fileR, String fileS, int joinKeyR, int joinKeyS, Integer[] listResR, Integer[] listResS)throws IOException, URISyntaxException{
        String splitFlag = "\\|";
        // read file R and create HashTable.
        BufferedReader instreamR = HDFSReader(fileR);
        String r;
        Map<String,List<String[]>> hashTableR = new HashMap<String,List<String[]>>();
        while ((r=instreamR.readLine())!=null) {
            List<String[]> listHashValuesR = new ArrayList<String[]>();
            String[] currentLineR = r.split(splitFlag);
            String currentJoinKeyName = currentLineR[joinKeyR];
            String[] valuesOfRes = new String[listResR.length];
            for(int i=0; i<listResR.length;i++){
                valuesOfRes[i] = currentLineR[listResR[i].intValue()];
            }
            // if this join key name has put in hash table, then add this res into its vlueList. 
            if(hashTableR.keySet().contains(currentJoinKeyName)){
                listHashValuesR = hashTableR.get(currentJoinKeyName);
            } 
            listHashValuesR.add(valuesOfRes);
            hashTableR.put(currentJoinKeyName, listHashValuesR);
        }

        // read file S and find join result.
        BufferedReader instreamS = HDFSReader(fileS);
        String s;
        while ((s=instreamS.readLine())!=null) {
            String[] currentLineS = s.split(splitFlag);
            String currentJoinKeyNameS = currentLineS[joinKeyS];
            // if current joinkey in the key set of hashTableR, add to result.
            if(hashTableR.keySet().contains(currentJoinKeyNameS)){
                String[] valuesOfResS = new String[listResS.length];
                for(int i=0; i<listResS.length;i++){
                    valuesOfResS[i] = currentLineS[listResS[i].intValue()];
                }
                for(int i=0; i<hashTableR.get(currentJoinKeyNameS).size();i++){
                    String[] res = new String[listResS.length + listResR.length+1];
                    String[] valuesOfResRi = hashTableR.get(currentJoinKeyNameS).get(i);
                    res[0] = currentJoinKeyNameS;
                    System.arraycopy(valuesOfResRi, 0, res, 1, valuesOfResRi.length);
                    System.arraycopy(valuesOfResS, 0, res, 1+valuesOfResRi.length, valuesOfResS.length);
                    result.add(res);
                }
            }
        }
    }

    /**
	 * create HBase table and delete if table exists, then write the results.
	 * @param tableName Hbase table name.
	 * @exception IOException On input error.
	 * @exception URISyntaxException throws when string could not be parsed as a URI reference.
	 * @see IOException
	 * @see URISyntaxException
	 */
    public void writeHBase(String tableName)throws IOException, URISyntaxException{
        Logger.getRootLogger().setLevel(Level.WARN);
        // get list of column key name
        List<String> listOfColumnKey = new ArrayList<String>();
        for(int i=0;i<selectListResR.length;i++) listOfColumnKey.add("R"+String.valueOf(selectListResR[i]));
        for(int i=0;i<selectListResS.length;i++) listOfColumnKey.add("S"+String.valueOf(selectListResS[i]));
        
        // create HBase table and delete if table exists.
        Configuration configuration = HBaseConfiguration.create();
        HBaseAdmin hAdmin = new HBaseAdmin(configuration);
        if (hAdmin.tableExists(tableName)) {
            System.out.println("Table already exists, delete it!");
            hAdmin.disableTable(TableName.valueOf(tableName));
		    hAdmin.deleteTable(TableName.valueOf(tableName));
        }
        HTableDescriptor htd = new HTableDescriptor(TableName.valueOf(tableName));
        HColumnDescriptor cf=new HColumnDescriptor(COLUME_FAMILY); 
        htd.addFamily(cf);
        hAdmin.createTable(htd);
        hAdmin.close();
        System.out.println("table "+tableName+ " created successfully");

        // put record
        HTable table = new HTable(configuration,tableName);
        String lastRowKey = null;
        int indexOfcurrRowKey = 0;
        if(result.size()>0){
            for(int i=0; i<result.size();i++){
                String[] currentRecord = result.get(i);
                String rowKey = currentRecord[0];
                String columnKeyIdx="";
                if(rowKey.equals(lastRowKey)){
                    indexOfcurrRowKey++;
                    columnKeyIdx = "."+String.valueOf(indexOfcurrRowKey);
                }
                else{
                    indexOfcurrRowKey=0;
                    columnKeyIdx = "";
                }
                // System.out.println(rowKey);
                Put put = new Put(rowKey.getBytes());
                for(int j=1;j<currentRecord.length;j++){
                    put.add(COLUME_FAMILY.getBytes(), (listOfColumnKey.get(j-1)+columnKeyIdx).getBytes(),currentRecord[j].getBytes());
                    table.put(put);
                }
                lastRowKey= rowKey;
            }
        }
        table.close();
        System.out.println("put successfully");
    }
   
    /**
	 * extract useful informations form args.
	 * @param argss the args input form terminal.
	 * @exception IOException On input error.
	 * @exception URISyntaxException throws when string could not be parsed as a URI reference.
	 * @see IOException
	 * @see URISyntaxException
	 */
    public void processArgs(String[] argss) throws IOException, URISyntaxException{
		String argStr0 = argss[0].replaceAll(" ","");
        String argStr1 = argss[1].replaceAll(" ","");
        String argStr2 = argss[2].replaceAll(" ","");
        String argStr3 = argss[3].replaceAll(" ","");
        // get the index of split point
        int indexOfEqual0 = argStr0.indexOf('=');
        int indexOfEqual1 = argStr1.indexOf('=');
        int indexOfColon3 = argStr3.indexOf(':');

        // get the right args, they are: 
        //     fileR,fileS,RJoinKey,SJoinKey,listResR,listResS
        hdfsFileR = HDFS_BASE_PATH + argStr0.substring(indexOfEqual0 + 1);
        hdfsFileS = HDFS_BASE_PATH + argStr1.substring(indexOfEqual1 + 1);
        hashJoinKeyR = Integer.parseInt(argStr2.substring(argStr2.indexOf('R')+1, argStr2.indexOf('=')));
        hashJoinKeyS = Integer.parseInt(argStr2.substring(argStr2.indexOf('S')+1));
        String[] indexOfRess = argStr3.substring(indexOfColon3+1).split(",");

        ArrayList<Integer> arrayListResR = new ArrayList<Integer>();
        ArrayList<Integer> arrayListResS = new ArrayList<Integer>();
        for(int i= 0; i<indexOfRess.length; i++){
            if(indexOfRess[i].charAt(0) == 'R') arrayListResR.add(Integer.parseInt(indexOfRess[i].substring(1)));
            else if(indexOfRess[i].charAt(0) == 'S') arrayListResS.add(Integer.parseInt(indexOfRess[i].substring(1)));
            else{
                System.out.println("Illegal char in arg 'res:R4,S5'!!! ");
			    System.exit(1);
            }
        } 
        selectListResR = arrayListResR.toArray(new Integer[arrayListResR.size()]);
        selectListResS = arrayListResS.toArray(new Integer[arrayListResS.size()]);
    }

    /**
	 * get args form terminal
	 * select columns by hash join and write the results into HBase.
	 */
    public static void main(String[] args) throws IOException, URISyntaxException{
		if (args.length != 4) {
			System.out.println("Usage: java Hw1GrpX R=<file 1> S=<file 2> join:R2=S3 res:R4,S5");
			System.exit(1);
		}
        Hw1Grp0 hw1 = new Hw1Grp0();
        hw1.processArgs(args);
        hw1.processHashJoin(hdfsFileR, hdfsFileS, hashJoinKeyR, hashJoinKeyS, selectListResR, selectListResS);
        hw1.writeHBase(TABLE_NAME);
		System.out.println("Process Done!");
    }
}
