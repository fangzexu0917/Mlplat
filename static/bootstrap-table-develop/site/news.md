---
layout: simple title: News description: News and announcements for all things Bootstrap Table, including new releases.
---

## Bootstrap Table 1.16.0

<span class="post-date">11 Feb 2020</span>

#### Core

- **New:** Added `buttonsOrder` option.
- **New:** Added `headerStyle` option.
- **New:** Added `showColumnsSearch` option.
- **New:** Added `serverSort` option.
- **New:** Added `unfiltered` parameter for `getData` method.
- **Update:** Updated `event` name to lowercase hyphen format for vue component.
- **Update:** Updated `es-AR` locale.
- **Update:** Updated the default classes of semantic theme.
- **Update:** Improved the `resize` problem with multiple tables.
- **Update:** Fixed `checkAll` event bug with sortable checkbox field.
- **Update:** Fixed `checkbox` and not-found td style errors.
- **Update:** Fixed `customSearch` return empty array bug.
- **Update:** Fixed column checkboxes not being disabled when using `toggleAll`.
- **Update:** Fixed `flat` not polyfilled error in vue cli3.
- **Update:** Fixed `height` and `border` not aligned bug.
- **Update:** Fixed `jqXHR` `undefined` error using custom ajax.
- **Update:** Fixed `pageSize` set to all bug with filter.
- **Update:** Fixed `refreshOptions` bug with radio and checkbox.
- **Update:** Fixed `removeAll` bug in the last page when sidePagination is server.
- **Update:** Fixed `search` not always trigger in IE11 bug.
- **Update:** Fixed `search` width `escape` bug.
- **Update:** Fixed `showColumns` cannot work of foundation theme.
- **Update:** Fixed `showFullscreen` bug when setting height.
- **Update:** Fixed `sort` cannot work after searching.
- **Update:** Fixed `sortable` style error when using `table-sm`.
- **Update:** Fixed `sortStable` not work bug.
- **Update:** Fixed `triggerSearch` not work bug.
- **Update:** Supported build cross all platforms.
- **Remove:** Removed `resetWidth` method and use `resetView` instead.

#### Extensions

- **New(cookie):** Added new options to get/set/delete the values by a custom function.
- **New(cookie):** Added save re-order and resize support.
- **New(filter-control):** Added `filterControlContainer` option.
- **New(filter-control):** Added `filterCustomSearch` option.
- **New(filter-control):** Added object and function support in `filterData` column option.
- **New(filter-control):** Added support for using sticky-header extension.
- **New(filter-control):** Added support comparisons search(<, >, <=, =<, >=, =>).
- **New(fixed-columns):** Added all themes support.
- **New(fixed-columns):** Added `fixedRightNumber` option.
- **New(fixed-columns):** Added support for using filter-control extension.
- **New(group-by):** Add `Array` support for `groupByField` option.
- **New(group-by):** Added `customSort` option support.
- **New(multiple-sort):** Added custom `sorter` support.
- **New(multiple-sort):** Added `multiSortStrictSort` option.
- **New(multiple-sort):** Added `multiSort` method.
- **New(print):** Added `printFormatter` data-attribute support.
- **New(reorder-columns):** Added `orderColumns` method.
- **New(reorder-rows):** Added `search` and `cardView` supported.
- **New(sticky-header):** Added support for all themes.
- **New(toolbar):** Added support for all themes.
- **New(reorder-rows):** Added `search` and `cardView` support.
- **Update(cookie):** Fixed cookie localeStorage not work bug with filter-control.
- **Update(cookie):** Fixed `minimumCountColumns` not working bug.
- **Update(cookie):** Improved `cookiesEnabled` to support ' in `data-attribute`.
- **Update(editable):** Fixed `formatter` bug if the column was edited.
- **Update(filter-control):** Fixed `hideUnusedSelectOptions` not work bug.
- **Update(filter-control):** Fixed filter not work bug with `undefined`.
- **Update(filter-control):** Fixed missing parameter of `resetSearch` and `filterDataType`.
- **Update(filter-control):** Fixed `search` with filter-control `search` bug.
- **Update(filter-control):** Fixed the `value` of select display error using editable.
- **Update(fixed-columns):** Fixed checkbox bug with fixed columns.
- **Update(fixed-columns):** Updated default value to `0` of `fixedNumber` option.
- **Update(group-by):** Improved `number` type support.
- **Update(group-by):** Fixed new table using modal bug.
- **Update(group-by):** Fixed `scrollTo` method using group-by.
- **Update(mobile):** Fixed input keyboard bug.
- **Update(multiple-sort):** Fixed not destroy bug.
- **Update(multiple-sort):** Fixed sort not work with `boolean` bug.
- **Update(print):** Improved to use `undefinedText` option.
- **Update(print):** Fixed IE11 not work bug.
- **Update(reorder-columns):** Fixed detail view column reorder bug.
- **Update(resizable):** Fixed columns resizing not work bug.
- **Update(resizable):** Fixed not work via JavaScript.
- **Update(sticky-header):** Fixed not work bug with fullscreen.
- **Update(treegrid):** Fixed `virtualScroll` option bug.
- **Remove:** Removed natural-sorting extension.

## Bootstrap Table 1.15.5

<span class="post-date">12 Oct 2019</span>

- **New:** Added `jqXHR` for `responseHandler` option and `onLoadSuccess` event.
- **New:** Added `stickyHeaderOffsetLeft` and `stickyHeaderOffsetRight` for sticky-header.
- **New:** Added Serbian RS cyrillic and latin locales.
- **Update:** Improved `export` button when there is only one type.
- **Update:** Fixed column events click error with `detailView`.
- **Update:** Fixed bug for `searchOnEnterKey` and `showSearchButton` are true.
- **Update:** Fixed `onScrollBody` event and added parameter.
- **Update:** Fixed search input size bug with `iconSize` option.
- **Update:** Fixed filter control select cannot work more than one table.
- **Update:** Fixed virtual scroll to top error when using `append` method.
- **Update:** Fixed `events` cannot work on virtual scroll.
- **Update:** Fixed bottom border bug with `height` option.
- **Update:** Fixed min version throw cannot convert object to primitive value error.

## Bootstrap Table 1.15.4

<span class="post-date">13 Aug 2019</span>

- **New:** Added `query` to `queryParams` option.
- **New:** Added `filter` parameter of `customSearch` option.
- **Update:** Fixed search bug in hidden columns.
- **Update:** Fixed table zoom width calculating bug.
- **Update:** Fixed events of column formatted by nested table.
- **Update:** Fixed checkbox style display bug.
- **Update:** Fixed stack overflow error of `checkBy` method.
- **Update:** Fixed `showSearchButton` and `showSearchClearButton` style bug.
- **Update:** Fixed filter-control select `null` value handle error.
- **Update:** Fixed `showSearchClearButton` bug in filter-control extension.
- **Update:** Fixed `print` button appears twice bug.

## Bootstrap Table 1.15.3

<span class="post-date">11 Jul 2019</span>

- **New:** Added nl-BE, fr-CH and fr-LU locale.
- **Update:** Updated nl-NL, pt-BR, fr-BE, fr-FR, nl-BE and nl-NL locale.
- **Update:** Fixed treegrid duplicate rows bug.
- **Update:** Fixed `updateCellByUniqueId` method bug on a filtered table.
- **Update:** Fixed colspan group header display bug.
- **Update:** Fixed table footer display bug in some case.
- **Update:** Fixed `getOptions` bug.
- **Update:** Fixed `detailView` bug when hiding columns.
- **Update:** Fixed IE minify bug.
- **Update:** Fixed full screen scrolling bug.

## Bootstrap Table 1.15.2

<span class="post-date">24 Jun 2019</span>

#### Core

- **New:** Added `virtualScroll` and `virtualScrollItemHeight` options to support large data.
- **New:** Added vue component support.
- **New:** Added support comparisons search(<, >, <=, =<, >=, =>).
- **New:** Added `detailViewByClick` table option and `detailFormatter` column option.
- **New:** Added `showExtendedPagination` and `totalNotFilteredField` table options.
- **New:** Added `widthUnit` option to allow any unit.
- **New:** Added `multipleSelectRow` option to support ctrl and shift select.
- **New:** Added `onPostFooter`(`post-footer.bs.table`) event.
- **New:** Added `detailViewIcon` and `toggleDetailView` method to hide the show/hide icons.
- **New:** Added `showSearchButton` and `showSearchClearButton` options to improve the search.
- **New:** Added `showButtonIcons` and `showButtonText` options to improve the icons display.
- **New:** Added `visibleSearch` option search only on displayed/visible columns.
- **New:** Added `showColumnsToggleAll` option to toggle all columns.
- **New:** Added `cellStyle` to support checkbox field.
- **New:** Added checkbox and radio auto checked from html support.
- **New:** Added screen reader support for pagination.
- **New:** Added travis lint src and check docs scripts.
- **New:** Added webpack support and user rollup to build the src.
- **New:** Added a version number property.
- **New:** Improved `filterBy` method with `or` condition and custom filter algorithm.
- **New:** Improved `showColumn` and `hideColumn` methods with array of fields.
- **New:** Improved `scrollTo` method to allow `rows` units.
- **Update:** Rewrote all code to ES6.
- **Update:** Improved `pageList` options to support localization.
- **Update:** Improved the `totalRows` option.
- **Update:** Improved table footer.
- **Update:** Improved `getSelections` and `getAllSelections` methods.
- **Update:** Improved css frameworks themes.
- **Update:** Updated parameters of the `getData` method.
- **Update:** Updated parameters of the (un)checkAll events to `rowsAfter, rowsBefore`.
- **Update:** Updated parameters of the `updateRow` method to support `replace`.
- **Update:** Updated page number to 1 while making a server side sort.
- **Update:** Renamed table `maintainSelected` option to `maintainMetaData`.
- **Update:** Renamed method `refreshColumnTitle` to `updateColumnTitle`.
- **Update:** Fixed card view value to be aligned incorrectly bug.
- **Update:** Fixed `smartDisplay` option pagination bug.
- **Update:** Fixed data-* attribute is an object bug.
- **Update:** Fixed page separators click bug.
- **Update:** Fixed scrolling bug in IE11.
- **Update:** Fixed initHeader error caused by toggleColumn.
- **Update:** Fixed search input trigger multiple times bug.
- **Update:** Fix Pagination/totalRows not updated on `hideRow`.
- **Update:** Fixed columns title error.

#### Extensions

- **New(editable):** Added `onExportSaved` event.
- **New(export):** Added `forceExport` column option force export columns with hidden.
- **New(export):** Added function support of `fileName` option.
- **New(filter-control):** Added `filterDataCollector` to control the filter select options.
- **New(filter-control):** Added `filterOrderBy` and filterDefault column options.
- **New(multiple-sort):** Added bootstrap v4 theme support.
- **New(print):** Added RTL dir support.
- **Remove:** Removed group-by, multi-column-toggle, multiple-search, multiple-selection-row, select2-filter and
  tree-column extensions.
- **Update(cookie):** Fixed cookie search cannot work bug.
- **Update(editable):** Updated parameters of `onEditableSave` to `field, row, rowIndex, oldValue, $el`.
- **Update(editable):** Fixed editable rerender bug after saving data.
- **Update(export):** Updated to only export table header.
- **Update(export):** Fixed bug with the footer extensions while sorting.
- **Update(filter-control):** Added ability to handle boolean.
- **Update(filter-control):** Fixed DatePicker of filter-control does not work bug.
- **Update(filter-control):** Fixed clear filterControl with Cookie bug.
- **Update(filter-control):** Fixed loading screen with filter control.
- **Update(filter-control):** Fixed overwriting the searchText bug.
- **Update(filter-control):** Fixed filtering does not work json sub-object.
- **Update(filter-control):** Fixed select filter with formatter.
- **Update(multiple-sort):** Fixed multiple-sort does not work with data-query-params bug.
- **Update(page-jump-to):** Fixed `click` bug when paginationVAlign is 'both'.
- **Update(reorder-columns):** Fixed reorder columns cannot work bug.
- **Update(reorder-columns):** Fix search and columns bug after reorder columns.
- **Update(treegrid):** Fixed treegrid cannot work bug.

## Bootstrap Table 1.14.2

<span class="post-date">19 Mar 2019</span>

- **New(fixed-columns extension):** Added new version fixed-columns extension.
- **New(js):** Updated the style of loading message.
- **Update(js):** Updated refresh event params.
- **Update(locale):** Updated all locale translation with English as default.
- **Update(export extension):** Fixed export all rows to pdf bug.
- **Update(export extension):** Disabled export button when exportDataType is 'selected' and selection empty.
- **Update(addrbar extension):** Fixed addrbar extension remove hash from url bug.

## Bootstrap Table 1.14.1

<span class="post-date">5 Mar 2019</span>

- **New(css):** Added CSS Frameworks supported.
- **New(css):** Added [Semantic UI](http://semantic-ui.com) theme.
- **New(css):** Added [Bulma](http://bulma.io) theme.
- **New(css):** Added [Materialize](https://materializecss.com/) theme.
- **New(css):** Added [Foundation](https://foundation.zurb.com/) theme.
- **New(js):** Added data attribute support for `ignoreClickToSelectOn` option.
- **Update(js):** Fixed `detailView` find td elements bug.
- **Update(js):** Fixed `showColumns` close dropdown bug when item label clicking.
- **Update(js):** Fixed reset width error after `toggleFullscreen`.
- **Update(js):** Fixed `cardview` click event bug.

## Bootstrap Table 1.13.5

<span class="post-date">23 Feb 2019</span>

- **New(auto-refresh extension):** Rewrote auto-refresh extension to ES6.
- **Update(js):** Fixed showFullscreen cannot work bug.
- **Update(js):** Redefined customSearch option.
- **Update(js):** Fixed show footer cannot work bug.
- **Update(js):** Updated the parameter of `footerStyle`.
- **Update(js):** Added classes supported for `footerStyle`.
- **Update(js):** Fixed IE11 transform bug.
- **Update(js):** Removed beginning and end whitespace from td.
- **Update(export extension):** Fixed export selected bug.

## Bootstrap Table 1.13.4

<span class="post-date">05 Feb 2019</span>

- **New(sticky-header extension):** Rewrote sticky-header extension to ES6.
- **New(sticky-header extension):** Added to support bootstrap v4 and `theadClasses` option.
- **New(auto-refresh extension):** Icons update to font-awesome 5.
- **New(examples):** Added examples Algolia search.
- **Update(js):** Fixed `theadClasses` is not set when a `thead` exists.
- **Update(js):** Fixed table resize after mergeCell the first row.
- **Update(cookie extension):** Fixed cookie extension broken bug.
- **Update(cookie extension):** Fixed cookie extension unicode encode bug.
- **Update(package):** Added `sass` devDependencies.

## Bootstrap Table 1.13.3

<span class="post-date">28 Jan 2019</span>

- **New(js):** Supported full table classes of bootstrap v4.
- **New(css):** Rewrited bootstrap-table.css to scss.
- **New(accent-neutralise extension):** Rewrited accent-neutralise extension to ES6.
- **New(addrbar extension):** Rewrited addrbar extension to ES6 and supported attribute option.
- **New(group-by-v2 extension):** New `groupByFormatter` option.
- **New(pipeline extension):** New pipeline extension `bootstrap-table-pipeline`.
- **Remove(js):** Removed `striped` option and use classes instead.
- **Update(js):** Fixed `locale` option bug.
- **Update(js):** Fixed `sortClass` option bug.
- **Update(js):** Fixed `sortStable` option cannot work bug.
- **Update(js):** Improved built-in sort function and `customSort` logic.
- **Update(js):** Fixed horizontal scrollbar bug.
- **Update(cookie extension):** Improved cookie extension code.

## Bootstrap Table 1.13.2

<span class="post-date">18 Jan 2019</span>

- **New(js):** Added `paginationSuccessivelySize`, `paginationPagesBySide` and `paginationUseIntermediate` pagination
  options.
- **New(cookie extension):** Rewrited cookie extension to ES6.
- **New(cookie extension):** Saved `filterBy` method.
- **New(filter-control extension):** Added `placeholder` as a empty option to the select controls.
- **New(filter-control extension):** Added `clearFilterControl` method in order to clear all filter controls.
- **New(docs)** Added algolia search.
- **Update(js):** Fixed sort column shows hidden rows in `server` side pagination bug.
- **Update(js):** Fixed `scrollTo` bug.
- **Update(css):** Fixed no-bordered problem of bootstrap v4.
- **Update(filter-control extension):** Added bootstrap v4 icon support.

## New Website for Bootstrap v4

<span class="post-date">10 Jan 2019</span>

Bootstrap has released the latest version v4.2.1. Since Bootstrap Table has been mainly used for Bootstrap v3, We have
rebuilt the official documentation of Bootstrap Table while upgrading the code to Bootstrap v4.

Bootstrap Table Website is a fork of [Bootstrap](http://getbootstrap.com/).

### What’s new

Here are the highlights of what’s new and updated in new website.

- **New:** More detailed documentation.
- **New:** Added Extensions documentation.
- **New:** Supported for searching documentation.
- **New:** Added News Menu.
- **New:** Added New Examples for Bootstrap v4 instead v3.
- **Update:** Used new API display style instead table.
- **Remove:** Removed Translation Documentations.

## Bootstrap Table 1.13.1

<span class="post-date">01 Jan 2019</span>

- **New(js):** Added `theadClasses` option to supoort bootstrap v4.
- **New(js):** Updated the default icons to font-awesome 5.
- **New(locale):** Rewrited all locales to ES6.
- **New(editable extension):** Rewrited `bootstrap-table-editable` to ES6.
- **New(filter-control extension):** Rewrited `bootstrap-table-filter-control` to ES6.
- **New(treegrid extension):** Added `rootParentId` option.
- **Update(js):** Fixed `getHiddenRows` method bug.
- **Update(js):** Fixed `getOptions` method to remove data property.
- **Update(js):** Fixed no matches display error.
- **Update(js):** Fixed eslint warning and error.
- **Update(locale):** Improved `es-ES` locale.
- **Update(filter-control extension):** Fixed multiple choice bug.
- **Update(filter-control extension):** Fixed select all rows and `keyup` event error.
- **Update(export extension):** Fixed export in cardView display error.

## Bootstrap Table 1.13.0

<span class="post-date">27 Dec 2019</span>

- **New(js):** Rewrited bootstrap-table to ES6.
- **New(locale):** Added `fi-FI.js` locale.
- **New(build):** Used babel instead of grunt.
- **New(filter-control):** Added `created-controls.bs.table` event to filter-control.
- **New(export extension):** Rewrited export extension to ES6.
- **New(export extension):** Added export extension support bootstrap v4.
- **New(export extension):** Added `exportTable` method.
- **New(toolbar extension):** Rewrited toolbar extension to ES6.
- **New(toolbar extension):** Added toolbar extension supports bootstrap v4.
- **New(toolbar extension):** Added server sidePagination support
- **New(resizable extension):** Released new resizable extension version 2.0.0.
- **New(editable extension):** Allowed different x-editable configuration per table row.
- **New(addrbar extension):** Added addrbar extension.
- **Update(js):** Improved `check/uncheck` methods
- **Update(js):** Fixed cookie with `pageNumber` and `searchText` bug.
- **Update(js):** Fix `selections` bugs.
- **Update(js):** Added `customSearch` support data attribute.
- **Update(js):** Fixed can't search data with formatter.
- **Update(js):** Fixed `getRowByUniqueId` error when row unique id is undefined.
- **Update(js):** Fxied older bootstrap version bug.
- **Update(css):** Removed toolbar line-height.
- **Update(css):** Limitted fullscreen CSS rule scope.
- **Update(editable extension):** Fixed editable formatter bug.
- **Update(extension):** Fixed bug with export extension together.
- **Update(extension):** Removed click-edit-row and flat-json extensions.
