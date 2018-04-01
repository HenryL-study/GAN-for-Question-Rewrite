Dataset: ydata-yanswers-all-questions-v1_0

Set of questions and corresponding answers from Yahoo! Answers, 
version 1.0.


=====================================================================
This dataset is provided as part of the Yahoo! Research Alliance
Webscope program, to be used for approved non-commercial research
purposes by recipients who have signed a Data Sharing Agreement with
Yahoo!. This dataset is not to be redistributed. No personally
identifying information is available in this dataset. More information
about the Yahoo! Research Alliance Webscope program is available at
http://research.yahoo.com
=====================================================================


Full description:

Yahoo! Answers is a web site were people post questions and answers,
all of which are public to any web user willing to browse or download
them. The data we have collected is the Yahoo! Answers corpus as of 
10/25/2007. It includes all the questions and their corresponding answers.
The corpus distributed here contains 4483032 questions and their answers. 

In addition of question and answer text, the corpus contains a small
amount of meta data, i.e., which answer was selected as the best
answer, and the category and sub-category that was assigned to this
question. No personal information is included in the corpus. The
question URIs and all user ids were anonymized so that no identifying 
information is revealed.

This dataset may be used by researchers to learn and validate answer
extraction models. 


Content:

The question-and-answer corpus is stored in two parts

FullOct2007.xml.part1.gz
FullOct2007.xml.part2.gz

To reconstitute the corpus into a single file, FullOct2007.xml,
the following unix command line may be used:

zcat FullOct2007.xml.part1.gz FullOct2007.xml.part2.gz > FullOct2007.xml

The individual files are NOT valid parsable XML.  The separation is
guaranteed to be on a byte boundary, but not an XML stanza boundary.

WARNING: The complete, uncompressed corpus is 12GB.

The format for each question is the following:

- Each question is stored inside a "<vespaadd>" XML element.

- Each question has a unique anonymized URI stored inside the "<uri>"
  element. This element is mandatory.

- The subject of each question is stored inside a "<subject>"
  element. This element is mandatory.

- Each question may have an optional "<content>" element, which
  contains additional detail about the question stored in
  "<subject>". This element is optional.

- The answer selected as best is stored in the "<bestanswer>"
  element. The best answer is selected either by the asker, or by the
  participants in the thread through voting, if the asker did not
  select a best answer. This element is mandatory.

- All answers posted for this question are stored inside the
  "<nbestanswers>" element, using "<answer_item>" sub-elements. This
  element is mandatory.

- The question is optionally classified into the question taxonomy
  using three elements: "<maincat>" stores the main category of the
  question, e.g., "Travel"; "<cat>" stores the category of the
  question, e.g., "China"; "<subcat>" stores the sub-category of the
  question, e.g., "Asia Pacific". All these elements are optional.

- The id of the user who asked the question is stored in the <id> field.
  This element is mandatory.

- The id of the user who provided the best answer is in <best_id>.
  This element is mandatory.

- <qlang> and <language> indicate the language in which the question and
  the answer were posted. <qintl> indicates the location where this
  question was posted, e.g., US. These elements are optional.

- <date> indicates the date when the question was posted. <lastanswerts>
  is the time stamp of the last answer for this question. These elements
  are mandatory. <res_date> indicates the date when the questions was 
  resolved. <vot_date> indicates the date when the best answer was voted 
  (if any). The last two elements are optional.

References: 

M. Surdeanu, M. Ciaramita, H. Zaragoza. 
Learning to Rank Answers on Large Online QA Collections. 
Proc. of the 46th Annual Meeting of the Association of 
Computational Linguistics (ACL), 2008. 
