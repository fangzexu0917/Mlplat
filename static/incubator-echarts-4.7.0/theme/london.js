/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define(['exports', 'echarts'], factory);
    } else if (
        typeof exports === 'object' &&
        typeof exports.nodeName !== 'string'
    ) {
        // CommonJS
        factory(exports, require('echarts'));
    } else {
        // Browser globals
        factory({}, root.echarts);
    }
})(this, function (exports, echarts) {
    var log = function (msg) {
        if (typeof console !== 'undefined') {
            console && console.error && console.error(msg);
        }
    };
    if (!echarts) {
        log('ECharts is not Loaded');
        return;
    }

    var colorPalette = [
        '#02151a',
        '#043a47',
        '#087891',
        '#c8c8c8',
        '#b31d14',
        '#0b9cc1',
        '#f2f2f2',
        '#f07b75'
    ];

    var theme = {
        color: colorPalette,

        title: {
            textStyle: {
                fontWeight: 'normal',
                color: '#02151a'
            }
        },

        visualMap: {
            color: ['#02151a', '#a2d4e6']
        },

        toolbox: {
            color: ['#02151a', '#02151a', '#02151a', '#02151a']
        },

        tooltip: {
            backgroundColor: 'rgba(0,0,0,0.5)',
            axisPointer: {
                // Axis indicator, coordinate trigger effective
                type: 'line', // The default is a straight line： 'line' | 'shadow'
                lineStyle: {
                    // Straight line indicator style settings
                    color: '#02151a',
                    type: 'dashed'
                },
                crossStyle: {
                    color: '#02151a'
                },
                shadowStyle: {
                    // Shadow indicator style settings
                    color: 'rgba(200,200,200,0.3)'
                }
            }
        },

        // Area scaling controller
        dataZoom: {
            dataBackgroundColor: '#eee', // Data background color
            fillerColor: 'rgba(144,197,237,0.2)', // Fill the color
            handleColor: '#02151a' // Handle color
        },

        timeline: {
            lineStyle: {
                color: '#02151a'
            },
            controlStyle: {
                color: '#02151a',
                borderColor: '#02151a'
            }
        },

        candlestick: {
            itemStyle: {
                color: '#043a47',
                color0: '#087891'
            },
            lineStyle: {
                width: 1,
                color: '#b31d14',
                color0: '#c8c8c8'
            },
            areaStyle: {
                color: '#087891',
                color0: '#c8c8c8'
            }
        },

        map: {
            itemStyle: {
                color: '#ddd'
            },
            areaStyle: {
                color: '#087891'
            },
            label: {
                color: '#c12e34'
            }
        },

        graph: {
            itemStyle: {
                color: '#c12e34'
            },
            linkStyle: {
                color: '#02151a'
            }
        },

        gauge: {
            axisLine: {
                lineStyle: {
                    color: [
                        [0.2, '#043a47'],
                        [0.8, '#02151a'],
                        [1, '#b31d14']
                    ],
                    width: 8
                }
            }
        }
    };

    echarts.registerTheme('london', theme);
});
