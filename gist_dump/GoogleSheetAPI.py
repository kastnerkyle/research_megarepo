import os
import requests
from bs4 import BeautifulSoup


class GSheet(object):
    base_sheet_url = 'https://spreadsheets.google.com/feeds/worksheets/{0}/private/full'
    list_worksheet_url = 'https://spreadsheets.google.com/feeds/list/{0}/{{}}/private/full'
    cell_worksheet_url = 'https://spreadsheets.google.com/feeds/cells/{0}/{{}}/private/full'
    HTTPheaders = {'content-type': 'application/atom+xml'}

    def __init__(self, sheetID, user, password):
        #self.sheetID = sheetID
        loginUrl = 'https://www.google.com/accounts/ClientLogin?accountType=HOSTED_OR_GOOGLE&Email={0}&Passwd={1}&service=wise&source=SheetsTest'.format(user, password)

        headers = {'content-type': 'application/x-www-form-urlencoded'}
        loginResponse = requests.post(loginUrl, headers=headers).text.split('\n')[:-1]

        self.HTTPheaders['Authorization'] = 'GoogleLogin auth={}'.format(loginResponse[-1][5:])
        self.base_sheet_url = self.base_sheet_url.format(sheetID)
        self.list_worksheet_url = self.list_worksheet_url.format(sheetID)
        self.cell_worksheet_url = self.cell_worksheet_url.format(sheetID)

    def _validate_answer(self, r, code=requests.codes.created):
        if r.status_code != code:
            print r.status_code, r.reason
            print r.text
            exit()

    def getWorksheetID(self, name):
        r = requests.get(self.base_sheet_url, headers=self.HTTPheaders)

        sheetXML = BeautifulSoup(r.content)

        for worksheet in sheetXML.feed.findAll('entry'):
            if worksheet.title.string == name:
                return worksheet.id.string.split("/")[-1]

    def createWorksheet(self, name, header=[]):
        headerSize = len(header) if len(header) > 0 else 1

        payload = """
        <entry xmlns="http://www.w3.org/2005/Atom"
            xmlns:gs="http://schemas.google.com/spreadsheets/2006">
          <title>{0}</title>
          <gs:colCount>{1}</gs:colCount>
          <gs:rowCount>2</gs:rowCount>
        </entry>
        """

        r = requests.post(self.base_sheet_url, data=payload.format(name, headerSize), headers=self.HTTPheaders)

        self._validate_answer(r)

        worksheet = BeautifulSoup(r.content).entry
        worksheetId = worksheet.id.string.split("/")[-1]

        if len(header) > 0:
            self._addHeader(worksheetId, header)

        return worksheetId

    def _getCellVersion(self, worksheetID, c):
        url = self.cell_worksheet_url.format(worksheetID)
        r = requests.get(url + '/R1C{}'.format(c) , headers=self.HTTPheaders)
        return BeautifulSoup(r.content).findAll("link")[1].attrs['href'].split("/")[-1]

    def _addHeader(self, worksheetID, values):
        url = self.cell_worksheet_url.format(worksheetID)

        payload = """
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:batch="http://schemas.google.com/gdata/batch" xmlns:gs="http://schemas.google.com/spreadsheets/2006">
          <id>{0}</id>
        """
        for i, value in enumerate(values):
            payload += """
              <entry>
                <batch:id>A{column}</batch:id>
                <batch:operation type="update"/>
                <id>{{0}}/R1C{column}</id>
                <link rel="edit" type="application/atom+xml" href="{{0}}/R1C{column}/{version}"/>
                <gs:cell row="1" col="{column}" inputValue="{value}"/>
              </entry>
            """.format(column=i+1, value=value, version=self._getCellVersion(worksheetID, i+1))
        payload += """
        </feed>
        """

        # print payload.format(url)
        # exit()

        r = requests.post(url + '/batch', data=payload.format(url), headers=self.HTTPheaders)

        self._validate_answer(r, requests.codes.ok)

    def _getHeader(self, worksheetID):
        url = self.cell_worksheet_url.format(worksheetID)
        r = requests.get(url + '?max-row=1', headers=self.HTTPheaders)
        return [e.content.string.replace(' ', '').lower() for e in BeautifulSoup(r.content).feed.findAll("entry")]

    def addRow(self, worksheetID, values):
        sheetHeader = self._getHeader(worksheetID)

        payload = """
        <entry xmlns="http://www.w3.org/2005/Atom" xmlns:gsx="http://schemas.google.com/spreadsheets/2006/extended">
        """
        for i, value in enumerate(values):
            payload += """
              <gsx:{0}>{1}</gsx:{0}>
            """.format(sheetHeader[i], value)
        payload += """
        </entry>
        """

        r = requests.post(self.list_worksheet_url.format(worksheetID), data=payload, headers=self.HTTPheaders)

        self._validate_answer(r)

    def editCell(self, worksheetID, values):
        payload = """
        <entry xmlns="http://www.w3.org/2005/Atom" xmlns:gs="http://schemas.google.com/spreadsheets/2006">
          <id>https://spreadsheets.google.com/feeds/cells/key/worksheetId/private/full/R2C4</id>
          <link rel="edit" type="application/atom+xml" href="https://spreadsheets.google.com/feeds/cells/key/worksheetId/private/full/R2C4"/>
          <gs:cell row="1" col="1" inputValue="hurp"/>
        </entry>
        """

        r = requests.post(self.cell_worksheet_url.format(worksheetID), data=payload, headers=self.HTTPheaders)

        self._validate_answer(r)

    def __getWorksheetSortedByColumn(self, worksheetID, columnName):
        url = self.list_worksheet_url.format(worksheetID)
        r = requests.post(url + '?orderby=' + columnName, headers=self.HTTPheaders)
        self._validate_answer(r)
# sheetUID = '1wYmO4Rzd4595qE4sb0xNl8oLpXjoWmFsazoncUaIegE'
# sheet = GSheet(sheetUID, 'your.email@gmail.com', os.environ['gsheet_pass'])

# worksheetID = sheet.getWorksheetID("rwar43")
# sheet.addRow(worksheetID, ["9",3,1])
# sheet.addRow(worksheetID, ["a li78f4"," 521345 234"," 2345 234"])

# sheet.sortWorksheetByColumn(worksheetID, "wop")