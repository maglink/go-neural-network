package utils

import (
	"compress/gzip"
	"io"
	"net/http"
	"os"
)

func DownloadAndDecompress(url, filenameGz, filename string) error {
	if _, err := os.Stat(filename); !os.IsNotExist(err) {
		return nil
	}

	err := downloadFile(filenameGz, url)
	if err != nil {
		return err
	}

	err = decompress(filenameGz, filename)
	if err != nil {
		return err
	}

	return nil
}

func decompress(from, to string) error {
	fromFile, err := os.Open(from)
	if err != nil {
		return err
	}
	defer fromFile.Close()

	r, err := gzip.NewReader(fromFile)
	if err != nil {
		return err
	}
	defer r.Close()


	out, err := os.Create(to)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, r)
	return nil
}

func downloadFile(filepath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

