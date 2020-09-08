#pragma once

#include <QtGui>
#include <QGridLayout>
#include <QDockWidget>
#include <QGraphicsScene>
#include "StatisticsWidget.h"

class QStatisticsDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QStatisticsDockWidget(QWidget* pParent = 0);

private:
	QGridLayout			m_MainLayout;
	QStatisticsWidget	m_StatisticsWidget;
	QGraphicsScene scene;
};