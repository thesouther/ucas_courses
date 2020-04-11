/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified by Shimin Chen to demonstrate functionality for Homework 2
// April-May 2015

import java.io.IOException;
import java.util.StringTokenizer;
import java.util.ArrayList;
import java.util.List;
import java.io.DataInput;
import java.io.DataOutput;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

  /**
   * BDMS HW2: Map Reduce process
   * @author  CongCong
   * @version 1.0
   * @since   2020-04-10
	 */
public class Hw2Part1 {
  /**
	 * This is the Mapper class
   * reference: http://hadoop.apache.org/docs/r2.6.0/api/org/apache/hadoop/mapreduce/Mapper.html
	 * @param ikey type:Object
	 * @param ivalue type:Text
	 * @param okey type:Text
	 * @param ovalue type:TimeAndCountWritable
	 * @exception IOException On input error.
	 * @exception URISyntaxException throws when string could not be parsed as a URI reference.
	 * @see IOException
	 * @see URISyntaxException
	 */
  public static class TokenizerMapper 
       extends Mapper<Object, Text, Text, TimeAndCountWritable>{
    
    private final static DoubleWritable one = new DoubleWritable(1);
    private Text SDKey = new Text();
      
    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());
      
      if(itr.countTokens() == 3){
        String tempKey = itr.nextToken()+ " "+itr.nextToken();
        SDKey.set(tempKey); // set key

        Double time;
        try{
          time = Double.valueOf(itr.nextToken());
        }catch(Exception e) {
          return ;
        }
        TimeAndCountWritable timeAndCount = new TimeAndCountWritable(1,time);
        System.out.println("count:"+SDKey + timeAndCount.toString());
        context.write(SDKey, timeAndCount);
      }
    }
  }

  /**
	 * This is the Combiner class
	 * @param ikey type:Text
	 * @param ivalue type:TimeAndCountWritable
	 * @param okey type:Text
	 * @param ovalue type:TimeAndCountWritable
	 * @exception IOException On input error.
	 * @exception URISyntaxException throws when string could not be parsed as a URI reference.
	 * @see IOException
	 * @see URISyntaxException
	 */
  public static class TimeAndCountCombiner
       extends Reducer<Text,TimeAndCountWritable,Text,TimeAndCountWritable> {
    private TimeAndCountWritable result = new TimeAndCountWritable();

    public void reduce(Text key, Iterable<TimeAndCountWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      double sumTime = 0;
      int count = 0;

      for (TimeAndCountWritable val : values) {
        sumTime += val.getTime();
        count += val.getCount();
      }
      result.setCount(count);
      result.setTime(sumTime);
      context.write(key, result);
    }
  }

  /**
	 * This is the Reducer class
   * reference http://hadoop.apache.org/docs/r2.6.0/api/org/apache/hadoop/mapreduce/Reducer.html
   * We want to control the output format to look at the following:
   * <source> <destination> <count> <average time>
	 * @param ikey type:Text
	 * @param ivalue type:TimeAndCountWritable
	 * @param okey type:Text
	 * @param ovalue type:Text
	 * @exception IOException On input error.
	 * @exception URISyntaxException throws when string could not be parsed as a URI reference.
	 * @see IOException
	 * @see URISyntaxException
	*/
  public static class TimeAndCountReducer
       extends Reducer<Text,TimeAndCountWritable,Text,Text> {

    private Text result_key= new Text();
    private Text result_value= new Text();

    protected void setup(Context context) {
      // try {
      //   prefix= Text.encode("count of ").array();
      //   suffix= Text.encode(" =").array();
      // } catch (Exception e) {
      //   prefix = suffix = new byte[0];
      // }
    }

    public void reduce(Text key, Iterable<TimeAndCountWritable> values, 
                       Context context
                       ) throws IOException, InterruptedException {
      double sumTime = 0;
      int count = 0;
      for (TimeAndCountWritable val : values) {
        sumTime += val.getTime();
        count += val.getCount();
      }

      result_key.set(key);

      double avgTime = sumTime / count;
      result_value.set(String.format("%d %.3f", count,avgTime));

      context.write(result_key, result_value);
    }
  }

  /**
	 * BDMS HW2: Map Reduce process
	 * Usage: wordcount <in> [<in>...] <out>
	*/
  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length < 2) {
      // System.err.println("Usage: wordcount <in> [<in>...] <out>");
      System.exit(2);
    }

    Job job = Job.getInstance(conf, "Hw2 Part1");

    job.setJarByClass(Hw2Part1.class);

    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(TimeAndCountCombiner.class);
    job.setReducerClass(TimeAndCountReducer.class);

    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(TimeAndCountWritable.class);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);

    // add the input paths as given by command line
    for (int i = 0; i < otherArgs.length - 1; ++i) {
      FileInputFormat.addInputPath(job, new Path(otherArgs[i]));
    }

    // add the output path as given by the command line
    FileOutputFormat.setOutputPath(job,
      new Path(otherArgs[otherArgs.length - 1]));

    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }

  /**
	 * data type: TimeAndCountWritable
   * <IntWritable count, DoubleWritable time>
   * the record of count and time
	*/
  public static class TimeAndCountWritable implements Writable {
    private IntWritable count;
    private DoubleWritable time;

    public TimeAndCountWritable() {
      count = new IntWritable(0);
      time = new DoubleWritable(0);
    }

    public TimeAndCountWritable(int count, double time) {
      this.count = new IntWritable(count);
      this.time = new DoubleWritable(time);
    }

    public int getCount() {
      return count.get();
    }

    public double getTime() {
      return time.get();
    }

    public void setCount(int count) {
      this.count.set(count);
    }

    public void setTime(double time) {
      this.time.set(time);
    }

    public void readFields(DataInput in) throws IOException {
      count.readFields(in);
      time.readFields(in);
    }

    public void write(DataOutput out) throws IOException {
      count.write(out);
      time.write(out);
    }

    @Override
    public String toString() {
      return String.format("<%d, %.3f>", getCount(), getTime());
    }
  }
}
