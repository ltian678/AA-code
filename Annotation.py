class Element(object):
  def __init__(self, html_clean,level, element_id, element_parent_id, element_attributes, element_text,element_senti_attribute):
    self.source_text = html_clean
    self.level = level
    self.element_id = element_id
    self.element_parent_id = element_parent_id
    self.element_attributes = element_attributes
    self.element_text = element_text
    self.start_pos, self.end_pos = find_start_end_index(self.source_text,self.element_text)
    self.element_senti_attribute = element_senti_attribute

  def to_dict(self):
    return {
        'source_text': self.source_text,
        'level': self.level,
        'element_id':self.element_id,
        'element_parent_id': self.element_parent_id,
        'element_attributes': self.element_attributes,
        'element_text': self.element_text,
        'element_start_pos': self.start_pos,
        'element_end_pos': self.end_pos,
        'element_senti_attribute': self.element_senti_attribute,
    }





class Annotation(object):
  def __init__(self, file_path, txt_source, raw_annotation):
    self.file_path = file_path
    self.txt_source = txt_source
    self.raw_annotation = raw_annotation

  def to_dict(self):
    return {
        'file_path': self.file_path,
        'txt_source': self.txt_source,
        'raw_annotation': self.raw_annotation,
    }



class Appraisal_Anno(object):
  def __init__(self, start, end, entity):
    #start and end char position in the source text
    #entity is the appraisal label
    self.start = start
    self.end = end
    self.entity = entity

  def to_dict(self):
    return {
        'start': self.start,
        'end': self.end,
        'entity': self.entity,
    }
