System.register(["./chunk-vendor.js","./chunk-frameworks.js","./chunk-copy.js"],function(g){"use strict";var d,f,A,x,b,m,w,E,j,M,O;return{setters:[function(c){d=c.j,f=c.t,A=c.b,x=c.c,b=c.a1,m=c.a2},function(c){w=c.k,E=c.aE,j=c.aF,M=c.aG},function(c){O=c.c}],execute:function(){g({i:B,s:I});var c=Object.defineProperty,Q=Object.getOwnPropertyDescriptor,C=(e,t)=>c(e,"name",{value:t,configurable:!0}),P=(e,t,s,i)=>{for(var r=i>1?void 0:i?Q(t,s):t,n=e.length-1,o;n>=0;n--)(o=e[n])&&(r=(i?o(t,s,r):o(r))||r);return i&&r&&c(t,s,r),r};const Y=14,U=20,J=20,K=55;let p=class extends HTMLElement{constructor(){super(...arguments);this.smallDisplay=!1}connectedCallback(){this.classList.add("d-inline-flex")}get lastToken(){return this.tokens[this.tokens.length-1]}get text(){return this.tokens.map(e=>e.text).join("/")}get id(){return this.lastToken?this.lastToken.id:p.emptyScope.id}get type(){return this.lastToken?this.lastToken.type:p.emptyScope.type}get scope(){return this.hasScope()?{text:this.text,type:this.type,id:this.id,tokens:this.tokens}:p.emptyScope}set scope(e){this.renderTokens(e.tokens)}renderTokens(e){this.clearScope();let t=0,s=e.length;const i=this.smallDisplay?Y:J,r=this.smallDisplay?U:K;for(let a=e.length-1;a>=0&&!(t+Math.min(e[a].text.length,i)+5>r);a--)t+=Math.min(e[a].text.length,i)+5,s=a;const n=C(a=>m`${a.map(o)}`,"tokensTemplate"),o=C((a,v)=>{const $=a.text.length>i?`${a.text.substring(0,i-3)}...`:a.text;return m`
        <command-palette-token
          data-text="${a.text}"
          data-id="${a.id}"
          data-type="${a.type}"
          data-value="${a.value}"
          data-targets="command-palette-scope.tokens"
          hidden="${v<s}"
          class="color-fg-default text-semibold"
          style="white-space:nowrap;line-height:20px;"
          >${$}<span class="color-fg-subtle text-normal">&nbsp;&nbsp;/&nbsp;&nbsp;</span>
        </command-palette-token>
      `},"tokenTemplate");b(n(e),this),this.hidden=!this.hasScope(),s!==0&&(this.placeholder.hidden=!1)}removeToken(){this.lastToken&&(this.lastRemovedToken=this.lastToken,this.lastToken.remove(),this.renderTokens(this.tokens))}hasScope(){return this.tokens.length>0&&this.type&&this.id&&this.text}clearScope(){for(const e of this.tokens)e.remove();this.placeholder.hidden=!0}attributeChangedCallback(e,t,s){e==="data-small-display"&&t!==s&&this.renderTokens(this.tokens)}};C(p,"CommandPaletteScopeElement"),p.emptyScope={type:"",text:"",id:"",tokens:[]},P([d],p.prototype,"smallDisplay",2),P([f],p.prototype,"placeholder",2),P([A],p.prototype,"tokens",2),p=P([x],p);var N=Object.defineProperty,Z=Object.getOwnPropertyDescriptor,D=(e,t)=>N(e,"name",{value:t,configurable:!0}),y=(e,t,s,i)=>{for(var r=i>1?void 0:i?Z(t,s):t,n=e.length-1,o;n>=0;n--)(o=e[n])&&(r=(i?o(t,s,r):o(r))||r);return i&&r&&N(t,s,r),r};let l=g("C",class extends HTMLElement{constructor(){super(...arguments);this.defaultPriority=0}connectedCallback(){this.classList.add("py-2","border-top"),this.setAttribute("hidden","true"),this.renderElement("")}prepareForNewItems(){this.list.innerHTML="",this.setAttribute("hidden","true"),this.classList.contains("border-top")||this.classList.add("border-top")}hasItem(e){return this.list.querySelectorAll(`[data-item-id="${e.id}"]`).length>0}renderElement(e){b(D(()=>this.hasTitle?m`
          <div class="d-flex flex-justify-between my-2 px-3">
            <span data-target="command-palette-item-group.header" class="color-fg-muted text-bold f6 text-normal">
              ${this.groupTitle}
            </span>
            <span data-target="command-palette-item-group.header" class="color-fg-muted f6 text-normal">
              ${e?"":this.groupHint}
            </span>
          </div>
          <div
            role="listbox"
            class="list-style-none"
            data-target="command-palette-item-group.list"
            aria-label="${this.groupTitle} results"
          ></div>
        `:m`
          <div
            role="listbox"
            class="list-style-none"
            data-target="command-palette-item-group.list"
            aria-label="${this.groupTitle} results"
          ></div>
        `,"groupTemplate")(),this)}push(e){this.removeAttribute("hidden"),this.topGroup&&this.atLimit?e.itemId!==this.firstItem.itemId&&this.replaceTopGroupItem(e):this.list.append(e)}replaceTopGroupItem(e){this.list.replaceChild(e,this.firstItem)}groupLimitForScope(){const e=this.closest("command-palette");if(e){const t=e.query.scope.type;return JSON.parse(this.groupLimits)[t]}}get limit(){const e=this.groupLimitForScope();return this.topGroup?1:this.isModeActive()?50:e||l.defaultGroupLimit}isModeActive(){const e=this.closest("command-palette");return e?e.getMode():!1}get atLimit(){return this.list.children.length>=this.limit}get topGroup(){return this.groupId===l.topGroupId}get hasTitle(){return this.groupId!==l.footerGroupId}get itemNodes(){return this.list.querySelectorAll("command-palette-item")}get firstItem(){return this.itemNodes[0]}get lastItem(){return this.itemNodes[this.itemNodes.length-1]}});D(l,"CommandPaletteItemGroupElement"),l.defaultGroupLimit=5,l.topGroupId="top",l.defaultGroupId="default",l.footerGroupId="footer",l.helpGroupIds=["modes_help","filters_help"],l.commandGroupIds=["commands"],l.topGroupScoreThreshold=9,y([d],l.prototype,"groupTitle",2),y([d],l.prototype,"groupHint",2),y([d],l.prototype,"groupId",2),y([d],l.prototype,"groupLimits",2),y([d],l.prototype,"defaultPriority",2),y([f],l.prototype,"list",2),y([f],l.prototype,"header",2),l=g("C",y([x],l));var q=Object.defineProperty,V=(e,t)=>q(e,"name",{value:t,configurable:!0});class _{constructor(t,s,{scope:i,subjectId:r,subjectType:n,returnTo:o}={}){this.queryText=t,this.queryMode=s,this.scope=i!=null?i:p.emptyScope,this.subjectId=r,this.subjectType=n,this.returnTo=o}get text(){return this.queryText}get mode(){return this.queryMode}get path(){return this.buildPath(this)}buildPath(t,s=t.text){return`scope:${t.scope.type}-${t.scope.id}/mode:${t.mode}/query:${s}`}clearScope(){this.scope=p.emptyScope}hasScope(){return this.scope.id!==p.emptyScope.id}isBlank(){return this.text.trim().length===0}isPresent(){return!this.isBlank()}immutableCopy(){const t=this.text,s=this.mode,i={...this.scope};return new _(t,s,{scope:i,subjectId:this.subjectId,subjectType:this.subjectType,returnTo:this.returnTo})}hasSameScope(t){return this.scope.id===t.scope.id}params(){const t=new URLSearchParams;return this.isPresent()&&t.set("q",this.text),this.hasScope()&&t.set("scope",this.scope.id),this.mode&&t.set("mode",this.mode),this.returnTo&&t.set("return_to",this.returnTo),this.subjectId&&t.set("subject",this.subjectId),t}}g("Q",_),V(_,"Query");var F=Object.defineProperty,ee=Object.getOwnPropertyDescriptor,G=(e,t)=>F(e,"name",{value:t,configurable:!0}),u=(e,t,s,i)=>{for(var r=i>1?void 0:i?ee(t,s):t,n=e.length-1,o;n>=0;n--)(o=e[n])&&(r=(i?o(t,s,r):o(r))||r);return i&&r&&F(t,s,r),r};let h=class extends HTMLElement{constructor(){super(...arguments);this.itemId="",this.itemTitle="",this.subtitle="",this.selected=!1,this.score=0,this.titleNodes=[],this.rendered=!1,this.newTabOpened=!1,this.containerDataTarget="command-palette-item.containerElement",this.containerClasses="mx-2 px-2 rounded-2 d-flex flex-items-start no-underline border-0",this.containerStyle="padding-top: 10px; padding-bottom: 10px; cursor: pointer;"}renderOcticon(e){this.iconElement.innerHTML=e}renderAvatar(e,t){b(this.item.getAvatarTemplate(e,t),this.iconElement)}setItemAttributes(e){this.item=e,this.itemId=e.id,this.itemTitle=e.title,this.hint=e.hint,this.href=e.path,e.subtitle&&(this.subtitle=e.subtitle)}connectedCallback(){this.subtitle&&this.subtitleElement.removeAttribute("hidden"),this.titleNodes.length>0&&(this.titleElement.textContent="",this.titleElement.append(...this.titleNodes))}onClick(e){this.item.activate(this.commandPalette,e)}get commandPalette(){return this.closest("command-palette")}attributeChangedCallback(e,t,s){this.rendered&&(e==="data-selected"?(this.setSelectionAppearance(),this.item.select(this)):e==="data-item-title"?this.titleElement.textContent=s:e==="data-subtitle"&&(this.subtitleElement.textContent=s))}setSelectionAppearance(){this.selected?(this.containerElement.classList.add("color-bg-subtle"),this.hintText.hidden=!1):(this.containerElement.classList.remove("color-bg-subtle"),this.hintText.hidden=!0)}renderLinkContainer(e){return m`
      <a
        data-target="${this.containerDataTarget}"
        data-action="click:command-palette-item#onClick"
        href="${this.href}"
        class="${this.containerClasses}"
        data-skip-pjax
        style="${this.containerStyle}"
      >
        ${e}
      </a>
    `}renderSpanContainer(e){return m`
      <span
        data-target="${this.containerDataTarget}"
        class="${this.containerClasses}"
        style="${this.containerStyle}"
        data-action="click:command-palette-item#onClick"
      >
        ${e}
      </span>
    `}renderElementContent(){return m`
      <div
        data-target="command-palette-item.iconElement"
        class="mr-2 color-fg-muted d-flex flex-items-center"
        style="height: 24px;"
      ></div>

      <div class="flex-1 d-flex flex-column" style="line-height: 24px;">
        <span data-target="command-palette-item.titleElement" class="color-fg-default f5">${this.itemTitle}</span>
        <p data-target="command-palette-item.subtitleElement" class="color-fg-muted f6 mb-0" hidden>${this.subtitle}</p>
      </div>

      <div class="color-fg-muted f5" style="line-height: 20px;">
        <span class="hide-sm" data-target="command-palette-item.hintText" style="line-height: 24px;" hidden
          >${this.item.getHint()}</span
        >
        <span
          class="hide-sm"
          data-target="command-palette-item.persistentHint"
          style="line-height: 24px;"
          hidden
        ></span>
      </div>
    `}renderElement(){const e=this.renderElementContent();b(G(()=>this.href?this.renderLinkContainer(e):this.renderSpanContainer(e),"itemTemplate")(),this),this.rendered=!0}activateLinkBehavior(e,t,s){const i=this.containerElement;s&&i instanceof HTMLAnchorElement?(this.newTabOpened=!0,this.openLinkInNewTab(i)):(this.newTabOpened=!1,this.openLink(i))}openLinkInNewTab(e){const t=e.getAttribute("target");e.setAttribute("target","_blank"),e.click(),t?e.setAttribute("target",t):e.removeAttribute("target")}openLink(e){e.click()}copyToClipboardAndAnnounce(e,t){O(e);const s=this.hintText,i=s.textContent;s.classList.add("color-fg-success"),s.textContent=t!=null?t:"Copied!",setTimeout(()=>{s.classList.remove("color-fg-success"),s.textContent=i},2e3)}getHint(){return this.hint?m`<span class="hide-sm">${this.hint}</span>`:this.item.scope?m`<div class="hide-sm">
        <kbd class="hx_kbd">Enter</kbd>
        to jump to
        <kbd class="hx_kbd ml-1">Tab</kbd>
        to search
      </div>`:m`<span class="hide-sm">Jump to</span>`}};G(h,"CommandPaletteItemElement"),u([d],h.prototype,"itemId",2),u([d],h.prototype,"itemTitle",2),u([d],h.prototype,"subtitle",2),u([d],h.prototype,"selected",2),u([d],h.prototype,"score",2),u([f],h.prototype,"titleElement",2),u([f],h.prototype,"iconElement",2),u([f],h.prototype,"subtitleElement",2),u([f],h.prototype,"hintText",2),u([f],h.prototype,"persistentHint",2),u([f],h.prototype,"containerElement",2),h=u([x],h);var te=function(){for(var e=new Uint32Array(256),t=256;t--;){for(var s=t,i=8;i--;)s=s&1?3988292384^s>>>1:s>>>1;e[t]=s}return function(r){var n=-1;typeof r=="string"&&(r=function(v){for(var $=v.length,X=new Array($),L=-1;++L<$;)X[L]=v.charCodeAt(L);return X}(r));for(var o=0,a=r.length;o<a;o++)n=n>>>8^e[n&255^r[o]];return(n^-1)>>>0}}(),se=function(e){return e<0&&(e=4294967295+e+1),("0000000"+e.toString(16)).slice(-8)},ie=g("c",function(e,t){var s=te(e);return t?se(s):s}),re=Object.defineProperty,ne=(e,t)=>re(e,"name",{value:t,configurable:!0});function I(e,t){const s=document.querySelector("command-palette"),i={command_palette_session_id:s.sessionId,command_palette_scope:s.query.scope.type,command_palette_mode:s.getMode(),command_palette_title:(t==null?void 0:t.group)==="commands"?t.title:"",command_palette_position:t==null?void 0:t.position,command_palette_score:t==null?void 0:t.score,command_palette_group:t==null?void 0:t.group,command_palette_item_type:t==null?void 0:t.itemType};w(`command_palette_${e}`,i)}ne(I,"sendTrackingEvent");var oe=Object.defineProperty,H=(e,t)=>oe(e,"name",{value:t,configurable:!0});function B(e){T.register(e)}H(B,"item");const R=class{constructor(e){this.position="",this.newTabOpened=!1;var t;this.title=e.title,this.priority=e.priority,this.score=e.score,this.subtitle=e.subtitle,this.typeahead=e.typeahead,this.scope=e.scope,this.hint=e.hint,this.icon=e.icon,this.group=(t=e.group)!=null?t:l.defaultGroupId,this.match_fields=e.match_fields,this._action=e.action}static register(e){this.itemClasses[e.itemType]=e}static get itemType(){return this.buildItemType(this.name)}static buildItemType(e){return e.replace(/([A-Z]($|[a-z]))/g,"_$1").replace(/(^_|_Item$)/g,"").toLowerCase()}static build(e){const t=this.itemClasses[e.action.type];if(t)return new t(e);throw new Error(`No item handler for ${e.action.type}`)}get action(){return this._action}get element(){return this._element||(this._element=new h,this._element.setItemAttributes(this)),this._element}getAvatarTemplate(e,t){return m`<img src="${e}" alt="${t}" class="avatar avatar-1 avatar-small circle" />`}getHint(){return this.element.getHint()}get key(){return`${this.action.type}/${this.title}/${this.group}`}get id(){return this._id||(this._id=ie(this.key).toString()),this._id}get path(){return this.action.path||""}get itemType(){return R.buildItemType(this.constructor.name)}select(e){}deselect(e){}activate(e,t){I(this.activateTrackingEventType,this)}get activateTrackingEventType(){return"activate"}activateLinkBehavior(e,t,s){this.element.activateLinkBehavior(e,t,s)}copy(e){I("copy",this)}render(e,t){return this.element.renderElement(),e&&(this.element.selected=!0),t&&(this.element.titleNodes=this.emphasizeTextMatchingQuery(this.title,t)),this.element}emphasizeTextMatchingQuery(e,t){if(!E(t,e))return[document.createTextNode(e)];const s=[];let i=0;for(const n of j(t,e)){if(e.slice(i,n)!==""){const v=document.createTextNode(e.slice(i,n));s.push(v)}i=n+1;const a=document.createElement("strong");a.textContent=e[n],s.push(a)}const r=document.createTextNode(e.slice(i));return s.push(r),s}copyToClipboardAndAnnounce(e,t){this.element.copyToClipboardAndAnnounce(e,t)}calculateScore(e){const t=this.matchFields.map(s=>this.calculateScoreForField({field:s,queryText:e}));return Math.max(...t)}calculateScoreForField({field:e,queryText:t}){return E(t,e)?M(t,e):-1/0}get matchFields(){return this.match_fields?this.match_fields:[this.title]}autocomplete(e){const t=e.commandPaletteInput,s=this.typeahead;s!==void 0?t.value=t.overlay+s:t.value=t.overlay+this.title}};let T=g("I",R);H(T,"Item"),T.itemClasses={},T.defaultData={title:"",score:1,priority:1,action:{type:"",path:""},icon:{type:"octicon",id:"dash-color-fg-muted"},group:l.defaultGroupId};var ae=Object.defineProperty,le=(e,t)=>ae(e,"name",{value:t,configurable:!0});class z{fuzzyFilter(t,s,i=0){const r=[];for(const n of t)n.calculateScore(s.text)>i&&r.push(n);return r}}le(z,"ProviderBase");var ce=Object.defineProperty,he=(e,t)=>ce(e,"name",{value:t,configurable:!0});class S extends z{constructor(t){super();this.element=t}get type(){return this.element.type}get modes(){return this.element.modes}get debounce(){return this.element.debounce}get scopeTypes(){return this.element.scopeTypes}get src(){return this.element.src}get hasWildCard(){return this.element.hasWildCard}get hasCommands(){return this.element.hasCommands}fetch(t,s){throw new Error("Method not implemented.")}enabledFor(t){throw new Error("Method not implemented.")}clearCache(){throw new Error("Method not implemented.")}}g("S",S),he(S,"ServerDefinedProvider");var pe=Object.defineProperty,de=(e,t)=>pe(e,"name",{value:t,configurable:!0});class k extends S{fetch(t,s){return this.fetchSrc(t)}enabledFor(t){return this.modeMatch(t)&&this.scopeMatch(t)}clearCache(){}scopeMatch(t){return this.scopeTypes.length===0||this.scopeTypes.includes(t.scope.type)}modeMatch(t){return this.modes.includes(t.mode)||this.hasWildCard}async fetchSrc(t,s=""){var i;if(!this.src)throw new Error("No src defined");const r=new URL(this.src,window.location.origin),n=t.params();s&&n.set("mode",s),r.search=n.toString();const o=await fetch(r.toString(),{headers:{Accept:"application/json","X-Requested-With":"XMLHttpRequest"}});if(o.ok){const a=await o.json();return{results:((i=a.results)==null?void 0:i.map(v=>T.build(v)))||[],octicons:a.octicons}}else return{error:!0,results:[]}}}g("R",k),de(k,"RemoteProvider");var me=Object.defineProperty,ue=(e,t)=>me(e,"name",{value:t,configurable:!0});class W extends k{constructor(){super(...arguments);this.maxItems=1e3,this.scopedItems={}}clearCache(){super.clearCache(),this.scopedItems={}}get debounce(){return 0}async prefetch(t){if(this.scopedItems[t.scope.id])return;const s=new _("",t.mode,{subjectId:t.subjectId,subjectType:t.subjectType,returnTo:t.returnTo,scope:t.scope}),i=await this.fetchSrc(s,t.mode);this.octicons=i.octicons||[];const r=i.results||[];this.scopedItems[t.scope.id]=r}async fetch(t,s){const i=this.scopedItems[t.scope.id]||[];return t.isBlank()?{results:i.slice(0,this.maxItems)}:{results:this.fuzzyFilter(i,t).slice(0,this.maxItems)}}}g("P",W),ue(W,"PrefetchedProvider")}}});
//# sourceMappingURL=chunk-prefetched-provider-0edbd5ee.js.map