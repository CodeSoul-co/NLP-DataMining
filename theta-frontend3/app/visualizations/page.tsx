'use client';

import { useState, useEffect } from 'react';
import { Image, Loader2, Download, PieChart, BarChart3, TrendingUp } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { WorkspaceLayout } from '@/components/workspace-layout';
import { ETMAgentAPI, ResultInfo, VisualizationInfo } from '@/lib/api/etm-agent';

function VisualizationsContent() {
  const [results, setResults] = useState<ResultInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedResult, setSelectedResult] = useState<ResultInfo | null>(null);
  const [visualizations, setVisualizations] = useState<VisualizationInfo[]>([]);

  useEffect(() => {
    loadResults();
  }, []);

  const loadResults = async () => {
    try {
      setLoading(true);
      const data = await ETMAgentAPI.getResults();
      setResults(data);
      if (data.length > 0) {
        setSelectedResult(data[0]);
        await loadVisualizations(data[0].dataset, data[0].mode);
      }
    } catch (error) {
      console.error('Failed to load results:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadVisualizations = async (dataset: string, mode: string) => {
    try {
      const data = await ETMAgentAPI.getVisualizations(dataset, mode);
      setVisualizations(data);
    } catch (error) {
      console.error('Failed to load visualizations:', error);
      setVisualizations([]);
    }
  };

  const handleSelectResult = async (result: ResultInfo) => {
    setSelectedResult(result);
    await loadVisualizations(result.dataset, result.mode);
  };

  const getVisualizationIcon = (type: string) => {
    switch (type) {
      case 'bar':
        return <BarChart3 className="w-8 h-8 text-blue-600" />;
      case 'heatmap':
        return <TrendingUp className="w-8 h-8 text-orange-600" />;
      case 'image':
        return <Image className="w-8 h-8 text-green-600" />;
      default:
        return <PieChart className="w-8 h-8 text-purple-600" />;
    }
  };

  const getVisualizationName = (name: string) => {
    const names: Record<string, string> = {
      'topic_distribution': '主题分布',
      'word_cloud': '词云图',
      'topic_heatmap': '主题热力图',
    };
    return names[name] || name;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="p-8 h-full overflow-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-slate-900 mb-2">可视化图表</h2>
        <p className="text-slate-600">查看训练结果的可视化展示</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* 结果选择器 */}
        <div className="lg:col-span-1">
          <Card className="p-4 bg-white border border-slate-200">
            <h3 className="font-medium text-slate-900 mb-4">选择数据集</h3>
            {results.length === 0 ? (
              <div className="text-center text-slate-500 text-sm py-8">
                暂无结果，请先完成模型训练
              </div>
            ) : (
              <div className="space-y-2">
                {results.map((result) => (
                  <Card
                    key={`${result.dataset}-${result.mode}`}
                    className={`p-3 cursor-pointer transition-colors ${
                      selectedResult?.dataset === result.dataset &&
                      selectedResult?.mode === result.mode
                        ? 'bg-blue-50 border-blue-200'
                        : 'hover:bg-slate-50 border-slate-200'
                    }`}
                    onClick={() => handleSelectResult(result)}
                  >
                    <p className="font-medium text-slate-900 text-sm">
                      {result.dataset}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">{result.mode}</p>
                  </Card>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* 可视化图表 */}
        <div className="lg:col-span-3">
          {selectedResult ? (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {visualizations.length === 0 ? (
                <Card className="col-span-full p-8 bg-white border border-slate-200 text-center">
                  <Image className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                  <p className="text-slate-500">暂无可视化图表</p>
                </Card>
              ) : (
                visualizations.map((viz) => (
                  <Card
                    key={viz.name}
                    className="p-6 bg-white border border-slate-200 hover:shadow-lg transition-all cursor-pointer group"
                  >
                    <div className="aspect-video bg-slate-100 rounded-lg mb-4 flex items-center justify-center">
                      {getVisualizationIcon(viz.type)}
                    </div>
                    <h4 className="font-medium text-slate-900 mb-2">
                      {getVisualizationName(viz.name)}
                    </h4>
                    <p className="text-xs text-slate-500 mb-4">
                      类型: {viz.type}
                    </p>
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full gap-2 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <Download className="w-4 h-4" />
                      下载图表
                    </Button>
                  </Card>
                ))
              )}
            </div>
          ) : (
            <Card className="p-8 bg-white border border-slate-200 text-center">
              <PieChart className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-500">选择一个数据集查看可视化图表</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

export default function VisualizationsPage() {
  return (
    <WorkspaceLayout currentStep="visualizations">
      <VisualizationsContent />
    </WorkspaceLayout>
  );
}
