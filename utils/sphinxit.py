# sphinxit.py
"""Contains methods to help document project in sphinx."""

import pyexcel as pe
from typing import Tuple, Generator
from config.conf import logger


class Sphinxifier:

    def __init__(self):
        """
        Initialize Sphinxifier class.

            :param self:
        """
        pass

    @staticmethod
    def _generate_sheets_content(
        book: pe.Book, sheet_name: str
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Generate sheet rst string and the sheet name
        to be used as output file name.

            :param book:
            :param sheet_name:
        """
        if sheet_name:
            sheet = book[sheet_name]
            yield str(sheet.content), sheet.name
        else:
            for sheet in book:
                yield str(sheet.content), sheet.name

    def make_rst_table(self, excel_doc_path: str, sheet_name: str=None) -> None:
        """
        Make and print rst table to file.

            :param excel_doc_path: full or relative path to the
                excel sheet containing table to be converted
            :param sheet_name: give a sheet name if multiple sheets exist
                however this is optional when converting all sheets in
                excel workbook to rst format.
        """
        book = pe.get_book(file_name=excel_doc_path)

        gen_val = self._generate_sheets_content(book, sheet_name)

        for data, name in gen_val:

            logger.info(f"converting sheet - {name} to rst format.")

            with open(name + ".rst", 'w') as fp:
                fp.write(str(data))

        logger.info("All rst conversion completed!")
