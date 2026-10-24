# Vendored ECharts

- Package: `echarts@6.1.0`
- File: `echarts.min.js` (from `package/dist/echarts.min.js`)
- License: Apache-2.0 — see `LICENSE.echarts`
- Upstream: <https://github.com/apache/echarts>
- NOTICE: see `NOTICE.echarts`

The report page loads this local copy only. There is no CDN dependency, so the
page keeps working offline and stays within the `default-src 'self'` policy.

Update procedure:

```bash
npm pack echarts@<version>
tar -xzf echarts-<version>.tgz
cp package/dist/echarts.min.js web/report/vendor/echarts.min.js
cp package/LICENSE web/report/vendor/LICENSE.echarts
cp package/NOTICE web/report/vendor/NOTICE.echarts
```

Record the version and the license check in the commit message when updating.
