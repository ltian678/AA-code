
import pandas as pd
import xml.etree.ElementTree as et
import os
from web_anno_tsv import open_web_anno_tsv
import statistics
import re
import html
import logging
import sys
import json
from collections import Counter
import glob

from xml.etree.ElementTree import XML, fromstring
import xml.etree.ElementTree as ET
import re
from Annotation import Element, Annotation, Appraisal_Anno

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def read_xml_file(file_path):
  xml_file = open(file_path,'r')
  Lines = xml_file.readlines()
  print('len of Lines ',len(Lines))
  newLines = []
  for idx, l in enumerate(Lines):
    if ('<segment' not in l) and ('</segment>' not in l) and (len(l)>2):
      print('exception here! 0',l)
      print('idx ',idx)
      #print('length of l',len(l))
    else:
      newLines.append(l)
    #print('idx',idx)
  return newLines



def recursive_print(element, indent,pindent):
    element_tag = element.tag
    element_attributes = element.attrib['features'].split(';')[1] if len(element.attrib) else ""
    element_text = element.text if element.text is not None and len(element.text.strip()) > 0 else ""
    element_tail = element.tail.strip() if element.tail is not None and len(element.tail.strip()) > 0 else ""
    element_label = element.get('features')
    element_id = element.attrib['id'] if len(element.attrib) else ""
    element_parent_id = element.attrib['parent'] if (len(element.attrib) and 'parent' in element.attrib) else ""
    print(
        " "*indent,
        'level: '+str(indent/4) + ' ||',
        'element_id '+element_id,
        ' | parent '+element_parent_id,
        #'tag" '+element_tag,
        #'title: '+element_tag.title(),
        #'label: '+element_label,
        '|| Attribute: ',
        element_attributes,
        '|| text: '+element_text,
        #element_tail
    )

    element_children = list(element)
    #print('FIRST element children ',element_children[0])
    #print('FIRST children ', element_children[0].text)
    iid = 0
    for child in element_children:
        iid = recursive_print(child, indent + 4, pindent)
        pindent.append(indent+4)
    return pindent


def dig_polarity_label(input_lst):
    pol_label= 'No_senti'
    if len(input_lst):
      if 'positive' in input_lst:
        pol_label = 'positive'
      elif 'negative' in input_lst:
        pol_label = 'negative'
      elif 'neutral' in input_lst:
        pol_label = 'neutral'
    return pol_label



def recursive_access(html_clean,element, indent,element_lst):
    #print('element ',element)
    #print("element join nested text ","".join(element.itertext()))
    #print('T/F check on element.text', element.text is not None)
    #print('T/F check on th length of text strip ', len(element.text.strip()))
    element_tag = element.tag
    element_attributes = element.attrib['features'].split(';')[1] if len(element.attrib) else ""
    element_senti_attribute = dig_sentiment_label(element.attrib['features'].split(';')) if len(element.attrib) else ""
    #remove the T/F check on the element text for element.text and len(element.text.strip())
    element_text = "".join(element.itertext())
    element_tail = element.tail.strip() if element.tail is not None and len(element.tail.strip()) > 0 else ""
    element_label = element.get('features')
    element_id = element.attrib['id'] if len(element.attrib) else ""
    element_parent_id = element.attrib['parent'] if (len(element.attrib) and 'parent' in element.attrib) else ""


    element_obj = Element(html_clean,indent/4, element_id, element_parent_id, element_attributes, element_text,element_senti_attribute)
    #print('element_obj ', element_obj)
    element_children = list(element)
    element_lst.append(element_obj.to_dict())
    for child in element_children:
      element_object = recursive_access(html_clean,child, indent+4,element_lst)
    return element_lst



def read_str(input_str):
  logger.info(f"input_str {input_str}")
  #print('og input_str ', input_str)
  input_str = input_str.replace('\n','')
  clean_input = re.sub('<[^>]*>', '', input_str)
  html_clean = html.unescape(clean_input)
  #print('clean_input', clean_input)
  print('HTML clean input ', html_clean)
  logger.info(f"CLEAN text: {html_clean}")
  '''
  try:
    print('original input str', input_str)
    t = ET.ElementTree(ET.fromstring(input_str))
    start_point = 'segment'
  except:
  '''
  xmlstring = '<input>'+input_str+'</input>'
  #print('xmlstring ',xmlstring)
  t = ET.ElementTree(ET.fromstring(xmlstring))
  start_point = 'input'
  #all_seg = None
  #n = None
  #print('start point ', start_point)
  for n in t.iter(start_point):
    a = recursive_print(n,0,[])
    max_a = str(max(a) / 4)
    logger.info(f"Depth: {max_a}")

  file_handler = logging.FileHandler('logs4.log')
  logger.setLevel(logging.DEBUG)
  file_handler.setLevel(logging.DEBUG)
  logger.addHandler(file_handler)
  #return n,all_seg


def grab_str(input_str):
  input_str1 = input_str.replace('\n','')
  clean_input = re.sub('<[^>]*>', '', input_str1)
  html_clean = html.unescape(clean_input)
  xmlstring = '<input>'+input_str+'</input>'
  t = ET.ElementTree(ET.fromstring(xmlstring))
  start_point = 'input'
  #print('html_clean, ',html_clean)
  anno_lst = []
  for n in t.iter(start_point):
    a = recursive_access(html_clean,n,0,[])
    #print('a ',a)
    for aa in a:
      anno_lst.append(aa)
      #print('a.element_text',aa.element_text)
      #print('a.attibute ',aa.element_attributes)
      #print('a.start_pos ', aa.start_pos)
      #print('a.end_pos ', aa.end_pos)
  return html_clean,anno_lst


def gen_annotations(input_file):
  obj_lst = []
  Lines = read_file(input_file)
  for L in Lines:
    #print('L ',L)
    try:
      clean_txt, raw_annotation = grab_str(L)
      #update the raw_annotation object for level 1 correct element_text
      indexes = find_level_index(raw_annotation)
      updated_raw_annotation = reconstruct_level_text(raw_annotation, indexes)
      new_anno_obj = Annotation(input_file, clean_txt,updated_raw_annotation)
      obj_lst.append(new_anno_obj)
    except:
      print('L ',L)
  return obj_lst


def gen_annotations_v2(input_file):
  obj_lst = []
  Lines = read_file(input_file)
  for L in Lines:
    try:
      clean_txt, raw_annotation = grab_str(L)
      #now the element.text should be fix with the correct start and end posistion
      new_anno = Annotation(input_file, clean_txt, raw_annotation)
      obj_lst.append(new_anno)
    except:
      print('Error Here with L ',L)
  return obj_lst



def find_start_end_index(source_text, keyword):
  start_index = source_text.find(keyword)
  end_index = start_index + len(keyword)
  return start_index, end_index



def find_level_index(raw_annotation):
  indexes = [i for i,x in enumerate(raw_annotation) if x['level'] == 1.0]
  indexes.append(len(raw_annotation))
  return indexes

def reconstruct_level_text(raw_annotation, indexes):
  for idx,ii in enumerate(indexes):
    if idx < len(indexes)-1:
      end_pos = indexes[idx+1]
      #print('end_pos ',end_pos)
      new_lst = raw_annotation[ii:end_pos]
      max_end = []
      min_start = []
      for n in new_lst:
        if len(n['element_text']) > 0:
          min_start.append(n['element_start_pos'])
          max_end.append(n['element_end_pos'])
        #print('n.element_start_pos ',n['element_start_pos'])
        #print('n.element_end_pos ',n['element_end_pos'])
        #print('n.text',n['element_text'])
      start_pos = min(min_start)
      end_pos = max(max_end)
      #print('start_pos ',start_pos)
      #print('end_pos ',end_pos)
      final_text = n['source_text'][start_pos:end_pos]
      #print('final text ', n['source_text'][start_pos:end_pos])
      raw_annotation[ii]['element_text'] = final_text
  #print('raw_annotation ', raw_annotation)
  return raw_annotation




def data_summary(data_lst):
    all_count = 0
    total_level1 = []
    total_level2 = []
    total_level3 = []
    att_lst = []
    att_lst_level1 = []
    att_senti_lst = []
    att_lst_level3 = []
    lv1_sent_lst = []
    lv2_sent_lst = []
    lv3_sent_lst = []
    level_one_len = []
    level_two_len = []
    level_three_len = []
    level_lst = []
    level_one_samples = []
    level_two_samples = []
    level_three_samples = []
    lv1_pair_lst = []
    lv2_pair_lst = []
    lv3_pair_lst = []
    ann_counter = 0
    source_length_lst = []
    level1_length_lst = []
    level2_length_lst = []
    level3_length_lst = []

    for f in data_lst:
      df = pd.read_pickle(f)
      logger.info(f"input file  {f}")
      logger.info(f"file shape {df.shape[0]}")
      all_count += df.shape[0]
      for index, row in df.iterrows():
        ann = row['raw_annotation']
        row_level1 = 0
        row_level2 = 0
        row_level3 = 0
        for an in ann:
          level_lst.append(an['level'])
          att_senti_lst.append(an['element_senti_attribute'])
          if an['level'] == 0:
            ann_counter += 1
            source_length_lst.append(len(an['element_text'].split()))
          if an['level'] == 1.0:
            row_level1 += 1
            att_lst_level1.append(an['element_attributes'])
            lv1_sent_lst.append(an['element_senti_attribute'])
            level_one_token_len = an['element_end_pos'] - an['element_start_pos']
            level_one_len.append(level_one_token_len)
            level_one_samples.append(an['element_text'])
            lv1_pair_lst.append((an['element_attributes'],an['element_senti_attribute']))
            level1_length_lst.append(len(list(an['element_text'])))
          if an['level'] == 2.0:
            row_level2 += 1
            att_lst.append(an['element_attributes'])
            lv2_sent_lst.append(an['element_senti_attribute'])
            level_two_token_len = an['element_end_pos'] - an['element_start_pos']
            level_two_len.append(level_two_token_len)
            level_two_samples.append(an['element_text'])
            lv2_pair_lst.append((an['element_attributes'],an['element_senti_attribute']))
            level2_length_lst.append(len(list(an['element_text'])))
          if an['level'] == 3:
            row_level3 += 1
            att_lst_level3.append(an['element_attributes'])
            lv3_sent_lst.append(an['element_senti_attribute'])
            level_three_token_len = an['element_end_pos'] - an['element_start_pos']
            level_three_len.append(level_three_token_len)
            level_three_samples.append(an['element_text'])
            lv3_pair_lst.append((an['element_attributes'],an['element_senti_attribute']))
            level3_length_lst.append(len(list(an['element_text'])))

        total_level1.append(row_level1)
        total_level2.append(row_level2)
        total_level3.append(row_level3)
    avg_lv1 = statistics.mean(level_one_len)
    avg_lv2 = statistics.mean(level_two_len)
    avg_lv3 = statistics.mean(level_three_len)
    total_lv1 = len(total_level1)
    mean_total_lv1 = statistics.mean(total_level1)
    mean_total_lv2 = statistics.mean(total_level2)
    total_lv1_pol = len(lv1_sent_lst)
    total_lv2_pol = len(lv2_sent_lst)
    anno_sent = ann_counter
    avg_lv1_token = statistics.mean(level1_length_lst)
    avg_lv2_token = statistics.mean(level2_length_lst)
    avg_lv3_token = statistics.mean(level3_length_lst)

    summary_obj = {}
    summary_obj['total_sentence'] = all_count
    summary_obj['average_character_lv1'] = avg_lv1
    summary_obj['average_character_lv2'] = avg_lv2
    summary_obj['average_character_lv3'] = avg_lv3
    summary_obj['average_lv1'] = mean_total_lv1
    summary_obj['average_lv2'] = mean_total_lv2
    summary_obj['total_lv1_polarity'] = total_lv1_pol
    summary_obj['total_lv2_polarity'] = total_lv2_pol
    summary_obj['average_token_lv1'] = avg_lv1_token
    summary_obj['average_token_lv2'] = avg_lv2_token
    summary_obj['average_token_lv3'] = avg_lv3_token
    return summary_obj


def from_xml_to_df(xml_file):
    obj_lst = gen_annotations_v2(xml_file)
    df = pd.DataFrame.from_records([s.to_dict() for s in obj_lst])
    return df

def read_all_files(root_dir):
    root_files = []
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            root_files.append(os.path.join(path,name))
    return root_files


def read_tsv(tsv_file_path):
  tsv1 = file_path
  #file_head1 = tsv1.split('/')[-1]
  file_head2 = tsv_file_path.split('/')[-1]
  txt_lst = []
  raw_annotations = []
  ann_obj_lst = []


  with open_web_anno_tsv(tsv1) as f:
    for i, sentence in enumerate(f):
      if sentence:
        txt_lst.append(sentence.text)
        raw_annotations.append(sentence.annotations)
        outer_ann = {}
        for j, annotation in enumerate(sentence.annotations):
          ann = {}
          ann['text'] = annotation.text
          ann['label'] = annotation.label
          #ann['label_id'] = annotation.label_id
          ann['start_pos'] = annotation.start
          ann['stop_pos'] = annotation.stop
          #ann_obj_lst.append(ann)
          outer_ann[j] = ann
        ann_obj_lst.append(outer_ann)
  return file_head2, txt_lst, raw_annotations, ann_obj_lst



# This script operates directly on the "annotation" folder output by exporting a WebAnno project
# for SOCC, this folder is SOCC\annotated\Appraisal\curation
# Each of \annotation's sub-folders contains a TSV that contains the annotations for the given comment.
# This script puts all of those TSVs into one long file, appending one after the other. In that file,
# commented lines using '#' indicate when the source TSVs begin and end.

import os
from smart_open import smart_open
import re

# path to a folder containing only the WebAnno TSVs
projectpath = '/content/drive/MyDrive/SFU_OpinionData/clean_TSVs'
# directory to output
outputpath = "/content/drive/MyDrive/SFU_OpinionData/combo_clean.tsv"

# get the subfolders of /curation
folders = os.listdir(projectpath)
# since all TSVs should be named CURATION_USER.tsv, we need to record the folder name to know which comment is being annotated.
# I use an embedded list for this.
files = [[f, os.listdir(os.path.join(projectpath, f))] for f in folders]
# so for each file 'f' in files, f[0] is the folder that f is contained in, and f[1] is the name of f
# check that each folder contains exactly one CURATION_USER.tsv file
if any([len(f[1]) for f in files]) > 1:
    bad_folders = [f[0] for f in files if len(f[1]) > 1]
    raise Exception('Some folders have more than one file:', bad_folders)
else:
    # since they have exactly one entry each, there's no point in keeping the filename in a list
    files = [[f[0], f[1][0]] for f in files]
    # check that that file is CURATION_USER.tsv
    if any([f[1] != 'CURATION_USER.tsv' for f in files]):
        bad_names = [f[1] for f in files if f[1] != 'CURATION_USER.tsv']
        raise Exception('Expected files named CURATION_USER.tsv; unexpected file names found:', bad_names)
        for f in files:
            if f != 'CURATION_USER.tsv':
                print(f)
    else:
        print('Found curated annotations')

# start combining the files
verbose = False     # setting this to True may help troubleshooting
newfile = ''
for f in files:
    name = f[0]
    f_path = os.path.join(projectpath, f[0], f[1])
    # indicate the beginning and end of a comment, and what that comment's name is
    newfile = newfile + '#comment: ' + name + '\n'
    with smart_open(f_path, 'r', encoding='utf-8') as f_io:
        newfile = newfile + f_io.read() + '#end of comment\n\n'
    if verbose:
        print('processed', name)

# output
print('All files processed, writing to', outputpath)
with smart_open(outputpath, 'w') as output:
        output.write(newfile)
print('Finished writing.')



def extract_id(id_str):
  #token_start = id_str.split('-')[0]
  token_id = id_str.split('-')[1]
  return token_id

def read_tsv(file_path):
  tsv1 = file_path
  #file_head1 = tsv1.split('/')[-1]
  file_head2 = tsv1.split('/')[-1]
  txt_lst = []
  raw_annotations = []
  ann_obj_lst = []
  span_lst = []


  with open_web_anno_tsv(tsv1) as f:
    for i, sentence in enumerate(f):
      if sentence:
        print('sentence', sentence)
        print('sentence dir ',dir(sentence))
        txt_lst.append(sentence.text)
        raw_annotations.append(sentence.annotations)
        print('sentence span length ', len(sentence.tokens))
        print('length of annotation ',len(sentence.annotations))
        for span in sentence.tokens:
          span_obj = {}
          span_obj['text'] = span.text
          span_obj['token_id'] = span.id.split('-')[1]
          span_obj['is_token'] = span.is_token
          span_obj['char_start'] = span.start
          span_obj['char_end'] = span.end
          span_lst.append(span)
        outer_ann = {}
        for j, annotation in enumerate(sentence.annotations):
          print('annotation ',annotation)
          for s in span_lst:
            if (s['char_start'] == annotation.start) and (s['char_end']==annotation.end) and (s['text']==annotation.text):
              s['annotation_label'] = annotation.label
          ann = {}
          ann['text'] = annotation.text
          ann['label'] = annotation.label
          ann['start_pos'] = annotation.start
          ann['stop_pos'] = annotation.stop
          #ann_obj_lst.append(ann)
          outer_ann[j] = ann
        ann_obj_lst.append(outer_ann)
  return file_head2, txt_lst, raw_annotations, ann_obj_lst



class annotation(object):
    def __init__(self, file_head2, txt_lst, raw_annotations, ann_obj_lst):
        #self.file_head1 = file_head1
        self.file_head2 = file_head2
        self.txt_lst = txt_lst
        self.raw_annotations = raw_annotations
        self.ann_obj_lst = ann_obj_lst

    def to_dict(self):
        return {
            #'file_head1': self.file_head1,
            'file_head2': self.file_head2,
            'txt_lst': self.txt_lst,
            'raw_annotations': self.raw_annotations,
            'ann_obj_lst': self.ann_obj_lst,
        }

def read_sample_tsv(sample_tsv_path):
    with open(sample_tsv_path) as f:
        for i, sentence in enumerate(f):
            print(f"Sentence {i}:", sentence.text)
            for j, annotation in enumerate(sentence.annotations):
                print(f"\tAnnotation {j}: ")
                print(f"\t\t Text: ", annotation.text)
                print(f"\t\t Label: ", annotation.label)
                print(f"\t\t Offsets: ", f"{annotation.start},{annotation.stop}")


def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def clean_app_label(row):
  app_label = row['app_label']
  most_comm = app_label
  if '|' in str(app_label):
    app_labels = app_label.split('|')
    clean_app_values = []
    seg_num =  "".join(app_labels[0].split('[')[1:])
    for app in app_labels:
      clean_app_values.append(app.split('[')[0])
    most_comm = Most_Common(clean_app_values)+"["+seg_num
  return most_comm

def clean_polarity_label(row):
  pol_label = row['polarity_label']
  most_comm_p = pol_label
  if '|' in str(pol_label):
    pol_labels = pol_label.split('|')
    clean_pol_values = []
    seg_num =  "".join(pol_labels[0].split('[')[1:])
    for pol in pol_labels:
      clean_pol_values.append(pol.split('[')[0])
    most_comm_p = Most_Common(clean_pol_values)+"["+seg_num
  return most_comm_p


def extract_sentence_id(row):
  return row['word'].split('-')[0]


def find_conflict_app_label(row):
  res = 0
  if '|' in str(row['app_label']):
    sep_labels = row['app_label'].split('|')
    clean_labels = []
    for s in sep_labels:
      clean_label = s.split('[')[0]
      clean_labels.append(clean_label)
    if len(set(clean_labels)) == 1:
      res = 0
    else:
      res = 1
  return res


def find_conflict_pol_label(row):
  res = 0
  if '|' in str(row['polarity_label']):
    sep_labels = row['polarity_label'].split('|')
    clean_labels = []
    for s in sep_labels:
      clean_label = s.split('[')[0]
      clean_labels.append(clean_label)
    if len(set(clean_labels)) == 1:
      res = 0
    else:
      res = 1
  return res

def find_num_app_label(row):
  res = 0
  if '|' in str(row['app_label']):
    sep_labels = row['app_label'].split('|')
    res = len(sep_labels)
  return res


def dataframe_conflict_handler(df):
    df['app_conflict'] = df.apply(lambda row: find_conflict_app_label(row),axis=1)
    df['pol_conflict'] = df.apply(lambda row: find_conflict_pol_label(row),axis=1)
    df['app_conflict_num'] = df.apply(lambda row: find_num_app_label(row),axis=1)
    return df


def data_statistics(root_folder):
    all_clean_tsv = glob.glob(root_folder+'/*.tsv')
    print('total tsv files ',len(all_clean_tsv))
    cc_files_major = []
    cc_files_minor = []
    app_labels_count = []
    for tsv_file in all_clean_tsv:
      df = pd.read_csv(tsv_file,sep='\t',names=['word','character_pos','text','app_label','polarity_label','add'])
      df['app_conflict'] = df.apply(lambda row: find_conflict_app_label(row),axis=1)
      df['pol_conflict'] = df.apply(lambda row: find_conflict_pol_label(row),axis=1)
      df['app_conflict_num'] = df.apply(lambda row: find_num_app_label(row),axis=1)
      app_lst = df.app_conflict.tolist()
      pol_lst = df.pol_conflict.tolist()
      app_count_lst = df.app_conflict_num.tolist()
      count_num_conflict_app = app_count_lst.count(3)

      if 1 in app_lst:
        count_c1 = app_lst.count(1)
        if count_c1 > 5:
          print('app label conflicts major ', tsv_file)
          cc_files_major.append(tsv_file)
        else:
          print('app label conflicts minor ', tsv_file)
          cc_files_minor.append(tsv_file)
          if count_num_conflict_app > 0:
            app_labels_count.append(tsv_file)
      if 1 in pol_lst:
        count_c2 = pol_lst.count(1)
        if count_c2 > 5:
          print('polarity label conflicts major ', tsv_file)
          cc_files_major.append(tsv_file)
        else:
          print('polarity label conflicts minor ', tsv_file)
          cc_files_minor.append(tsv_file)



    cc_ff = cc_files_major + cc_files_minor
    return cc_ff


def get_app_counter(df):
  app_l = df.AB_split.tolist()
  cleanedList = []
  for a in app_l:
    if str(a) != 'nan':
      if a[0] != '_':
        cleanedList.append(a)
  #cleanedList = [x for x in app_l if (str(x) != 'nan')]
  app_lst = [item for sublist in cleanedList for item in sublist]
  app_cc = Counter(app_lst)
  return app_cc


def split_sentence_id(row):
  if len(row['word'].split('-')) == 2:
    return row['word'].split('-')[0]

def split_token_id(row):
  if len(row['word'].split('-')) == 2:
    return row['word'].split('-')[1]

def split_app_label(row):
  if type(row['AB_split']) == list:
    return row['AB_split'][0]
  else:
    return row['AB_split']

def split_ab(row):
  if str(row['app_label']) != 'nan':
    return row['app_label'].split('|')
  else:
    return ['_']


def split_sent(row):
  if str(row['polarity_label']) != 'nan':
    return row['polarity_label'].split('|')
  else:
    return ['_']

def split_sent_label(row):
  if type(row['BA_split']) == list:
    return row['BA_split'][0]
  else:
    return row['BA_split']

def combo_label(row):
  return row['final_app_label'] + '-' + row['final_sent_label']

def handle_csv(file_path):
  df = pd.read_csv(file_path,sep='\t',names=['word','character_pos','text','app_label','polarity_label','add','add1'])
  #df['AB_split'] = df['app_label'].str.split('|')
  df['AB_split'] = df.apply(lambda row: split_ab(row),axis=1)
  df['BA_split'] = df.apply(lambda row: split_sent(row),axis=1)
  df['sentence_id'] = df.apply(lambda row: split_sentence_id(row),axis=1)
  df['token_id'] = df.apply(lambda row: split_token_id(row),axis=1)
  df['final_app_label'] = df.apply(lambda row: split_app_label(row),axis=1)
  df['final_sent_label'] = df.apply(lambda row: split_sent_label(row),axis=1)
  df['final_combo_label'] = df.apply(lambda row: combo_label(row),axis=1)
  df['file_path'] = file_path
  final_app_set = list(set(df.final_app_label.tolist()))
  final_app_counter = Counter(df.final_app_label.tolist())
  final_sent_set = list(set(df.final_sent_label.tolist()))
  final_combo_set = list(set(df.final_combo_label.tolist()))
  return df, final_combo_set, final_app_set, final_sent_set, final_app_counter


def count_label(app_lst):
  final_lst = []
  set_lst = []
  for a in app_lst:
    if (str(a) != 'nan') and (str(a)!= '_'):
      if '[' in a:
        set_lst.append(a)
      else:
        final_lst.append(a)
  overall_lst = final_lst + list(set(set_lst))
  return overall_lst

def count_combo_label(combo_lst):
  final_lst = []
  set_lst = []
  for c in combo_lst:
    if str(c) != '_-_':
      if '[' in c:
        set_lst.append(c)
      else:
        final_lst.append(c)
  o_lst = final_lst + list(set(set_lst))
  return o_lst


def new_handle_csv(file_path):
  df = pd.read_csv(file_path,sep='\t',names=['word','character_pos','text','app_label','polarity_label','add','add1'])
  #df['AB_split'] = df['app_label'].str.split('|')
  df['AB_split'] = df.apply(lambda row: split_ab(row),axis=1)
  df['BA_split'] = df.apply(lambda row: split_sent(row),axis=1)
  df['sentence_id'] = df.apply(lambda row: split_sentence_id(row),axis=1)
  df['token_id'] = df.apply(lambda row: split_token_id(row),axis=1)
  df['final_app_label'] = df.apply(lambda row: split_app_label(row),axis=1)
  df['final_sent_label'] = df.apply(lambda row: split_sent_label(row),axis=1)
  df['final_combo_label'] = df.apply(lambda row: combo_label(row),axis=1)
  df['file_path'] = file_path
  final_app_lst = count_label(df.final_app_label.tolist())
  final_sent_lst = count_label(df.final_sent_label.tolist())
  final_combo_lst = count_combo_label(df.final_combo_label.tolist())
  return df, final_app_lst, final_sent_lst, final_combo_lst


def combine_as_one_df(root_lst):
    final_app_labels = []
    final_sent_labels = []
    final_app_counter_lst = []
    final_combo_lst = []
    final_df_lst = []
    for f in root_lst:
      df, final_app, final_sent, final_combo = new_handle_csv(f)
      final_app_labels.append(final_app)
      final_sent_labels.append(final_sent)
      final_combo_lst.append(final_combo)
      final_df_lst.append(df)
    whole_df = pd.concat(final_df_lst)
    return whole_df



"""#Now we will prepare BIO input data for SFU dataset"""

#NOW we will include function to convert existing data to BIO tagging ready data
def prepare_dataset(sentence,annotations,sen_num):
  lst = {}
  keyorder = sentence.split()
  for i in annotations:
    sen = sentence[i['start']:i['end']].split()
    if len(sen) > 1:
      for l in range(len(sen)):
        if l == 0:
          lst[sen[l]] = 'B-'+i['entity']
        elif l == len(sen)-1:
          lst[sen[l]] = 'E-'+i['entity']
        else:
          lst[sen[l]] ='I-'+i['entity']

    else:
      lst["".join(sen)] = i['entity']
  for word in sentence.split():
    if word not in lst.keys() and word+' ' not in lst.keys():
      lst[word] = 'O'
  #print('lst ',lst)
  lst = sorted(lst.items(), key=lambda i:keyorder.index(i[0]))
  anno_lst = []
  for k in keyorder:
    if type(lst[k]) != list:
      anno_lst.append(lst[k])
    else:
      return None, None, None, None
  og_sentence = " ".join(keyorder)
  anno_str = " ".join(anno_lst)
  return lst, keyorder, og_sentence, anno_str


def gen_files_path(whole_df):
  return list(set(whole_df.file_path.tolist()))

def gen_small_df(whole_df, file_path):
  small_df = whole_df[whole_df['file_path']==file_path]
  small_df['text_str'] = small_df['text'].astype(str)
  new_small_df = small_df[small_df['text_str']!='nan']
  return new_small_df

def gen_label_span(new_test_df):
  label_span = {}
  for index, row in new_test_df.iterrows():
    f = row['final_app_label']
    if f != '_':
      if '[' in f:
        label_span[index] = f
  return label_span




def gen_final_txt_label_dct(new_test_df):
  #label_span = {}
  #new_test_labels = new_test_df.final_app_label.tolist()
  #for f in new_test_labels:
  #  if f != '_':
  #    if '[' in f:
  #      duplicate_label_times = new_test_labels.count(f)
  #      label_span[f] = duplicate_label_times
  #print('label_span', label_span)
  label_span = gen_label_span(new_test_df)
  print('here is the NEW label_span ',label_span)
  final_dict = {}
  label_span_counter = {}
  for index, row in new_test_df.iterrows():
    text =  row['text']
    text_label = row['final_app_label']
    obj_id = text + '_' + str(index)
    if text_label == '_':
      final_dict[obj_id] = 'O'
    else:
      if text_label in label_span.values():
        #t_label = label_span[index]
        label_span_lst = list(label_span.values())
        span_length = label_span_lst.count(text_label)
        #print('text_label ',text_label)
        #print('label_span_lst ', label_span_lst)
        #print('(text_label not in label_span_lst)',(text_label not in label_span_lst))
        if text_label in label_span_lst: #this situation we need to handle when text_label in this span_lst [with more than one tokens in this span]
          start_label = 'B-'+text_label
          in_label = 'I-'+text_label
          end_label = 'I-'+text_label
          if start_label not in final_dict.values():
            final_dict[obj_id] = start_label
            label_span_counter[text_label] = 1
          else:
            if label_span_counter[text_label] == (span_length-1):
              final_dict[obj_id] = end_label
            else:
              final_dict[obj_id] = in_label
              label_span_counter[text_label] += 1
      else:
        final_dict[obj_id] = 'B-'+text_label
        '''
        if (text_label not in label_span_lst):
          final_dict[obj_id] = 'B-'+ text_label
          if text_label not in label_span_counter.keys():
            label_span_counter[text_label] = 1
          else:
            label_span_counter[text_label] += 1
        else:
          if label_span_counter[text_label] == (span_length-1):
            final_dict[obj_id] = 'E-' + text_label
          else:
            final_dict[obj_id] = 'I-' + text_label
            label_span_counter[text_label] += 1
      else:
        final_dict[obj_id] = text_label
        '''
  return final_dict, label_span


def clean_label(row):
  og_label = row['og_label']
  clean_label = og_label
  if '[' in og_label:
    clean_label = og_label.split('[')[0]
  return clean_label


def clean_text(row):
  og_text = row['og_text']
  clean_text = og_text
  if '_' in og_text:
    clean_text = og_text.split('_')[0]
  return clean_text


def run(source_data_file, base_dir):
  whole_df = pd.read_pickle(source_data_file)
  file_paths = gen_files_path(whole_df)
  #base_dir to store BIO ready data
  txt_lst = []
  labels_lst = []
  for ff in file_paths:
    small_df = gen_small_df(whole_df, ff)
    final_dict, label_span = gen_final_txt_label_dct(small_df)
    label_df = pd.DataFrame(final_dict.items(), columns=['og_text', 'og_label'])
    label_df['label'] = label_df.apply(lambda row: clean_label(row),axis=1)
    label_df['text'] = label_df.apply(lambda row: clean_text(row),axis=1)
    label_df.to_pickle(base_dir+ff.split('/')[-2].split('.')[0]+'.pkl')
    txt_ll = label_df.text.tolist()
    label_ll = label_df.label.tolist()
    txt = ' '.join(txt_ll)
    labels = ' '.join(label_ll)
    txt_lst.append(txt)
    labels_lst.append(labels)
  return txt_lst, labels_lst



def gen_final_dict(df):
    final_dict = {}
    label_span_counter = {}
    for index, row in df.iterrows():
      text =  row['text']
      text_label = row['final_app_label']
      if text_label == '_':
        final_dict[text] = 'O'
      else:
        if text_label in label_span.keys():
          span_length = label_span[text_label]
          if (text_label not in final_dict.values()) and (('B-'+text_label) not in final_dict.values()):
            final_dict[text] = 'B-'+ text_label
            if text_label not in label_span_counter.keys():
              label_span_counter[text_label] = 1
            else:
              label_span_counter[text_label] += 1
          else:
            if label_span_counter[text_label] == (span_length-1):
              final_dict[text] = 'E-' + text_label
            else:
              final_dict[text] = 'I-' + text_label
              label_span_counter[text_label] += 1
        else:
          final_dict[text] = text_label
    return final_dict


"""#POST BIO Data Generator"""

'''
#Here should be the English input data files
english_lst = ['/content/Appraisal_Data/dataframe_v3/UOW2017_Tweets.pkl',
  '/content/Appraisal_Data/dataframe_v3/UOW2018_Blogs.pkl',
  '/content/Appraisal_Data/dataframe_v3/UOW2018_Tweets.pkl',
  '/content/Appraisal_Data/dataframe_v3/UOW2019.pkl',
  '/content/Appraisal_Data/dataframe_v3/UOWPH2122021.pkl',
  '/content/Appraisal_Data/dataframe_v3/UOWPH2092021.pkl',
  '/content/Appraisal_Data/dataframe_v3/UOWPH2012022.pkl']



chn_lst =  ['/content/Appraisal_Data/dataframe_v3/UOW2017_CMN_1.pkl',
  '/content/Appraisal_Data/dataframe_v3/UOW2017_CMN_2.pkl']

'''


def prepare_entity(row):
  entity_lst = []
  raw_annotations = row['raw_annotation']
  for r in raw_annotations:
    if (r['level'] == 2) and (r['element_attributes'] != 'no_attitude'):
      anno = Appraisal_Anno(r['element_start_pos'], r['element_end_pos'],r['element_attributes'])
      #entity = r['element_attributes']
      #start = r['element_start_pos']
      #end = r['element_end_pos']
      entity_lst.append(anno)
      #entity_lst.append((start,end,entity))
  return entity_lst


#NOW we will include function to convert existing data to BIO tagging ready data
def prepare_dataset(sentence,annotations,sen_num):
  lst = {}
  keyorder = sentence.split()
  for i in annotations:
    print('annotations', type(annotations))
    print('i', i.start)
    print('ii ',i.end)
    sen = sentence[i.start:i.end].split()
    if len(sen) > 1:
      for l in range(len(sen)):
        if l == 0:
          lst[sen[l]] = 'B-'+i.entity
        elif l == len(sen)-1:
          lst[sen[l]] = 'I-'+i.entity
        else:
          lst[sen[l]] ='I-'+i.entity

    else:
      lst["".join(sen)] = 'B-'+i.entity
  #print('sentence ', sentence)
  #print('lst',list(lst.keys()))
  #print('lst items',list(lst.items()))
  for word in sentence.split():
    if word not in lst.keys() and word+' ' not in lst.keys():
      lst[word] = 'O'
  #print('lst ',lst)
  #lst = sorted(lst.items(), key=lambda i:keyorder.index(i[0]))
  lst = sorted(lst.items())
  lst = [('sent'+str(sen_num),) + i for i in lst]
  return lst







def gen_final_txt_label_dct(new_test_df):
  #label_span = {}
  #new_test_labels = new_test_df.final_app_label.tolist()
  #for f in new_test_labels:
  #  if f != '_':
  #    if '[' in f:
  #      duplicate_label_times = new_test_labels.count(f)
  #      label_span[f] = duplicate_label_times
  #print('label_span', label_span)
  label_span = gen_label_span(new_test_df)
  print('here is the NEW label_span ',label_span)
  final_dict = {}
  label_span_counter = {}
  for index, row in new_test_df.iterrows():
    text =  row['text']
    text_label = row['final_app_label']
    obj_id = text + '_' + str(index)
    if text_label == '_':
      final_dict[obj_id] = 'O'
    else:
      if text_label in label_span.values():
        #t_label = label_span[index]
        label_span_lst = list(label_span.values())
        span_length = label_span_lst.count(text_label)
        #print('text_label ',text_label)
        #print('label_span_lst ', label_span_lst)
        #print('(text_label not in label_span_lst)',(text_label not in label_span_lst))
        if text_label in label_span_lst: #this situation we need to handle when text_label in this span_lst [with more than one tokens in this span]
          start_label = 'B-'+text_label
          in_label = 'I-'+text_label
          end_label = 'I-'+text_label
          if start_label not in final_dict.values():
            final_dict[obj_id] = start_label
            label_span_counter[text_label] = 1
          else:
            if label_span_counter[text_label] == (span_length-1):
              final_dict[obj_id] = end_label
            else:
              final_dict[obj_id] = in_label
              label_span_counter[text_label] += 1
      else:
        final_dict[obj_id] = 'B-'+text_label
        '''
        if (text_label not in label_span_lst):
          final_dict[obj_id] = 'B-'+ text_label
          if text_label not in label_span_counter.keys():
            label_span_counter[text_label] = 1
          else:
            label_span_counter[text_label] += 1
        else:
          if label_span_counter[text_label] == (span_length-1):
            final_dict[obj_id] = 'E-' + text_label
          else:
            final_dict[obj_id] = 'I-' + text_label
            label_span_counter[text_label] += 1
      else:
        final_dict[obj_id] = text_label
        '''
  return final_dict, label_span




def prepare_dataset_v2(sentence,annotations,sen_num):
  lst = {}
  #clean_s = re.sub('[^A-Za-z0-9]+', '', sentence)
  keyorder = sentence.split()
  '''
  for ik,k in enumerate(keyorder):
    print('ik ',ik)
    print('k ',k)
    find_k = int(sentence.find(k))
    end_k = int(find_k + len(k))
    print('res for find k', find_k)
    print('res for end_k ',end_k)
    print('found token, ', sentence[find_k:end_k])
  '''
  #print('keyorder ',keyorder)
  for i in annotations:
    #print('annotations', type(annotations))
    #print('i', i.start)
    #print('ii ',i.end)
    sen = sentence[i.start:i.end].split()
    #print('sen ',sen)
    #print('len sen',len(sen))

    if len(sen) >= 1:
      for l in range(len(sen)):
        #print('sen[l] ',sen[l])
        #print('check sen[l]', sen[0] in keyorder)
        if l == 0:
          lst[sen[l]] = 'B-'+i.entity
        elif l == len(sen)-1:
          lst[sen[l]] = 'I-'+i.entity
        else:
          lst[sen[l]] ='I-'+i.entity
    else:
      #print('check sen ', sen in keyorder)
      lst["".join(sen)] = 'B-'+i.entity
      #print('Triggered here! ', lst)
      #print('lst keys',lst.keys())
      #print('lst values ',lst.values())
  #print('sentence ', sentence)
  #print('lst',list(lst.keys()))
  #print('lst items',list(lst.items()))
  anno_lst = []
  for word in sentence.split():
    if word not in lst.keys() and word+' ' not in lst.keys():
      lst[word] = 'O'
  for w in sentence.split():
    # may need to
    #print('w ',w)
    #print('lst[w]',lst[w])
    clean_w = re.sub('[^A-Za-z0-9]+', '', w)
    #print('clean_w ',clean_w)
    if clean_w in lst.keys():
    #if w in lst.keys():
      #print('wwww' ,clean_w)
      w_label = lst[clean_w]
    elif w in lst.keys():
      #print('i should hit here!')
      #print('CLEAN_W', clean_w)
      w_label = lst[w]
    else:
      w_label = lst[w]
    anno_lst.append(w_label)
  #print('anno_lst HEERRRRR', anno_lst)
  #print('lst ',lst)
  #print('lst keys ',lst.keys())
  '''
  index_map = {v: i for i, v in enumerate(keyorder)}
  new_map = {}
  #print('index_map ',index_map)
  for kk in index_map.keys():
    if kk in keyorder:
      new_map[kk] = index_map[kk]
    elif kk.replace('.',"") in keyorder:

      new_map[kk] = index_map[kk]
  '''
  #print(sorted(lst.items(), key=lambda pair: index_map[pair[0]]))
  #lst = sorted(lst.items(), key=lambda i:keyorder.index(i[0]))
  #lst = sorted(lst.items())
  #lst = [('sent'+str(sen_num),) + i for i in lst]
  anno_str = ",".join(anno_lst)
  return sentence, anno_str
