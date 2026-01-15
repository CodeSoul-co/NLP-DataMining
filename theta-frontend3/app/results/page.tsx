'use client';

import { useState, useEffect } from 'react';
import { BarChart3, Loader2, ExternalLink, Download } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { WorkspaceLayout } from '@/components/workspace-layout';
import { ETMAgentAPI, ResultInfo } from '@/lib/api/etm-agent';

function ResultsContent() {
  const [results, setResults] = useState<ResultInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedResult, setSelectedResult] = useState<ResultInfo | null>(null);
  const [topicWords, setTopicWords] = useState<Record<string, string[]> | null>(null);
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);

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
        await loadResultDetails(data[0].dataset, data[0].mode);
      }
    } catch (error) {
      console.error('Failed to load results:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadResultDetails = async (dataset: string, mode: string) => {
    try {
      const [words, metricsData] = await Promise.all([
        ETMAgentAPI.getTopicWords(dataset, mode),
        ETMAgentAPI.getMetrics(dataset, mode).catch(() => null),
      ]);
      setTopicWords(words);
      setMetrics(metricsData);
    } catch (error) {
      console.error('Failed to load result details:', error);
      setTopicWords(null);
      setMetrics(null);
    }
  };

  const handleSelectResult = async (result: ResultInfo) => {
    setSelectedResult(result);
    await loadResultDetails(result.dataset, result.mode);
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
        <h2 className="text-2xl font-semibold text-slate-900 mb-2">分析结果</h2>
        <p className="text-slate-600">查看训练完成的主题模型分析结果</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 结果列表 */}
        <div className="lg:col-span-1 space-y-4">
          <Card className="p-4 bg-white border border-slate-200">
            <h3 className="font-medium text-slate-900 mb-4">结果列表</h3>
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
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <p className="font-medium text-slate-900 text-sm">
                          {result.dataset}
                        </p>
                        <p className="text-xs text-slate-500 mt-1">
                          {result.mode} · {result.num_topics} 主题
                        </p>
                        <p className="text-xs text-slate-400 mt-1">
                          {new Date(result.timestamp).toLocaleString()}
                        </p>
                      </div>
                      <BarChart3 className="w-5 h-5 text-blue-600" />
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* 结果详情 */}
        <div className="lg:col-span-2 space-y-6">
          {selectedResult ? (
            <>
              {/* 评估指标 */}
              {metrics && (
                <Card className="p-6 bg-white border border-slate-200">
                  <h3 className="font-medium text-slate-900 mb-4">评估指标</h3>
                  <div className="grid grid-cols-3 gap-4">
                    {Object.entries(metrics).map(([key, value]) => (
                      <div key={key} className="text-center p-4 bg-slate-50 rounded-lg">
                        <p className="text-2xl font-bold text-blue-600">
                          {typeof value === 'number' ? value.toFixed(3) : String(value)}
                        </p>
                        <p className="text-sm text-slate-500 capitalize">{key}</p>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* 主题词 */}
              {topicWords && (
                <Card className="p-6 bg-white border border-slate-200">
                  <h3 className="font-medium text-slate-900 mb-4">主题词分析</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(topicWords).map(([topic, words]) => (
                      <div key={topic} className="p-4 bg-slate-50 rounded-lg">
                        <p className="font-medium text-slate-900 mb-2 capitalize">
                          {topic.replace('_', ' ')}
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {words.map((word, index) => (
                            <span
                              key={index}
                              className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs"
                            >
                              {word}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* 操作按钮 */}
              <div className="flex gap-4">
                <Button variant="outline" className="gap-2">
                  <Download className="w-4 h-4" />
                  导出结果
                </Button>
                <Button variant="outline" className="gap-2">
                  <ExternalLink className="w-4 h-4" />
                  查看详细报告
                </Button>
              </div>
            </>
          ) : (
            <Card className="p-8 bg-white border border-slate-200 text-center">
              <BarChart3 className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-500">选择一个结果查看详情</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <WorkspaceLayout currentStep="results">
      <ResultsContent />
    </WorkspaceLayout>
  );
}
