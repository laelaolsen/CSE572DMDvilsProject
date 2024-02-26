# Starter code by M. Sneberger to find privacy Notices/Statements/Policies
# by recursively searching through HTML of a website and looking for strings
# that indicate the presence of a privacy Notice/Statement/Policies with the
# goal of scraping the text of those pages for possible NLP analysis.
#
# The 'example_input_test.csv' input file was derived by a version of this
# code that was used to identify websites that are subject to the privacy
# laws of the State of California. To begin a smaller 'mini_input_test.csv'
# is inserted for quick testing purposes. The idea behind taking input from
# .csv files and writing output to .csv files is twofold: 1) saves the
# previous and current output; and 2) allows offline filtering of both the
# input and the output.
#
# CURRENT LIMITATION with this code is it does not see 'privacy' on a website
# if that string is in a persistent footer.

from bs4 import BeautifulSoup
import requests
import csv
from urllib.parse import urljoin

urls = []
# the string 'do not sell' is facored to catch sites subject to California laws
strings = ['privacy', 'california privacy', 'california-privacy', 'california resident']

def explore(url, parent_url, depth):
 
    try:

        # base cases
        if url in urls:
            return 'False', url #Returns false if url has already been tried

        if not '.' in url:   # this is what handles relative paths
            url = parent_url + url  # libraries may be needed to deal with syntax
            print(parent_url, "added to", url)
        elif not url.startswith('https://') and not url.startswith('http://'):
            url = 'https://' + url  #if list does not have (this occurrs for all URLs in the list)
        if depth > 2:
            print("depth greater than 2")
            return 'False', url # often occurrs before false return, more than 2 recursions
        
        # current work
        try:
            page = requests.get(url, timeout=3)    # removed(proxies=proxy) - Minimize timer timer
        except:
            urls.append(url)
            return 'Req_Error', url                 # BS cannot reach site
        
        soup = BeautifulSoup(page.content, "html.parser")
        # for s in strings:
        #     found = soup.find(string=re.compile(s))
        #     if found:
        #         urls.append(url)
        #         return 'True', url, depth

        # Explore links in the main content
        links = soup.find_all('a')
        for link in links:
            for s in strings:
                if s in link.text.lower(): # try to eliminate case
                    #print(link)
                    absolute_url = urljoin(url, link.get('href'))
                    return 'True', absolute_url, depth #searches for the keywords in the actual text of the link, not the url

        # iteration, if the correct link is not found, try exploring all of the links on that page
        links = soup.find_all('a')
        for hit in links:
            #print(hit)
            next_url = hit.get('href')
            result = explore(next_url, url, depth + 1)  # recursive call
            if result[0]:
                return result

        urls.append(url)
        return 'False', url #Returns false if all links are exhausted with no match
    except Exception as error:
        if url != None:
            urls.append(url)
        print(url, error)
        return 'Gen_Error', url  # if found internal URL not a true URL

def csv_to_list(file_name):
    test_list_from_csv = []
    with open(file_name, mode ='r')as file:
   
        # reading the CSV file
        csvFile = csv.reader(file)
        next(csvFile)
    
        # move lines from vsc fiel to test_list_from_csv
        for line in csvFile:
            test_list_from_csv.append(line)
    return test_list_from_csv

def scrape_base():
    test_list_from_csv = csv_to_list('C:/Users/laela/Downloads/example_input_test.csv') #('example_input_test.csv') # for 1,924 URL input
    header = ['URL', 'String', 'String URL', 'Depth']

    #print(*test_list_from_csv, sep = '\n')
    result_list = []
    with open('example_results_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        counter = 1

        for line in test_list_from_csv:
            result = explore(line[0], None, 0)  # replace function as desired
            if result[0] == 'False' or result[0] == 'Req_Error' or result[0] == 'Gen_Error':    # filtering down results for future processing
                writer.writerow([line[0], result[0], result[1]])
            else:
                writer.writerow([line[0], result[0], result[1], result[2]])
            result_list.append(result[0])
            print(counter, line[0], result)      # to show progress in terminal
            counter += 1
    print(result_list)  # optional

def main():
    scrape_base()

if __name__ == "__main__":
    main()
