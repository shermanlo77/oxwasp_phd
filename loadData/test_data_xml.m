clc;
clearvars;
close all;

xml = xmlread('data/absBlock_noFilter_July16/scans/blank_36W/block36w_1.tif.profile.xml');
xml_image = xml.getDocumentElement();
xml_image.getElementsByTagName('kV').item(0).item(0).getWholeText
xml_image.getElementsByTagName('ExposureTime').item(0).item(0).getWholeText